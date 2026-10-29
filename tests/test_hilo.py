"""Run with `python -m pytest tests` or `python tests/test_hilo.py` (CPU, under a minute)."""
import copy
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hilo  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def test_hilonet_matches_reference():
    """10 steps of a trained HiloNet (parameters stored in the fixture) against the reference implementation."""
    fx = torch.load(FIX / "hilonet_ref.pt")
    net = hilo.HiloNet(hilo.HiloNetConfig(**fx["config"]), phi=fx["phi"], precision="fp32")
    ps = [w.clone().requires_grad_() for w in fx["W0"]]
    opt = hilo.Hilo(ps, total_steps=fx["T"], lr=fx["lr"], net=net)
    for gs in fx["grads"]:
        for p, g in zip(ps, gs):
            p.grad = g.clone()
        opt.step()
    for p, w in zip(ps, fx["W_final"]):
        assert torch.allclose(p.detach(), w, atol=1e-5), (p.detach() - w).abs().max()


def test_hilonet_random_init_save_load():
    cfg = hilo.HiloNetConfig(d=8, heads=2, mlp_hidden=16, tie=3, attn="softmax", betas=(0.9, 0.99), output_norm="rms")
    a, b = hilo.HiloNet(cfg, seed=1), hilo.HiloNet(cfg, seed=2)
    assert not torch.equal(a.flat(), b.flat()) and torch.equal(a.flat(), hilo.HiloNet(cfg, seed=1).flat())
    with tempfile.TemporaryDirectory() as d:
        a.save(Path(d) / "net.pt")
        c = hilo.Hilo([torch.zeros(2, 2)], total_steps=1, net=str(Path(d) / "net.pt")).net
    assert c.cfg == cfg and torch.equal(c.flat(), a.flat())
    p = torch.randn(12, 20, requires_grad=True)
    p.grad = torch.randn(12, 20)
    hilo.Hilo([p], total_steps=5, lr=0.1, net=c).step()
    assert torch.isfinite(p).all()


def test_output_norms():
    torch.manual_seed(1)
    for mode in hilo.OUTPUT_NORMS:
        net = hilo.HiloNet(hilo.HiloNetConfig(output_norm=mode))
        for shape in [(64, 16), (16, 64), (32, 32)]:
            p = torch.zeros(shape, requires_grad=True)
            p.grad = torch.randn(shape)
            hilo.Hilo([p], total_steps=10, lr=1.0, net=net).step()
            norm = p.detach().norm().item()
            if mode == "none":
                assert 0 < norm < float("inf")
            else:
                expect = hilo.norms.target_norm(mode, *shape)
                assert abs(norm - expect) < 1e-3 * expect, (mode, shape, norm, expect)
    x = torch.randn(3, 8, 4)
    assert torch.equal(hilo.normalize(x, "none"), x)
    assert torch.allclose(hilo.normalize(x, "rms").square().mean(dim=(1, 2)), torch.ones(3))


def test_weight_decay():
    """Decoupled: the run with weight decay differs from the one without by -lr * wd * W0."""
    torch.manual_seed(4)
    w0, g = torch.randn(6, 4), torch.randn(6, 4)
    out = []
    for wd in (0.0, 0.5):
        p = w0.clone().requires_grad_()
        p.grad = g.clone()
        hilo.Hilo([p], total_steps=10, lr=0.1, weight_decay=wd).step()
        out.append(p.detach())
    assert torch.allclose(out[1] - out[0], -0.1 * 0.5 * w0, atol=1e-6)


def test_assign_owners_balances_cost():
    from hilo.optim import assign_owners
    shapes = ([(768, 768)] * 4 + [(3072, 768), (768, 3072)]) * 12
    for world in (1, 3, 6, 8):
        owner = assign_owners(shapes, world)
        load = [0] * world
        for (m, n), r in zip(shapes, owner):
            load[r] += m * n * (m + n + 4096)
        assert max(load) <= 1.05 * sum(load) / world, (world, load)


def test_state_dict_roundtrip():
    torch.manual_seed(3)
    p = torch.randn(8, 4, requires_grad=True)
    opt = hilo.Hilo([p], lr=0.1, total_steps=10)
    p.grad = torch.randn(8, 4)
    opt.step()
    q = p.detach().clone().requires_grad_()
    opt2 = hilo.Hilo([q], lr=0.1, total_steps=10)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    g = torch.randn(8, 4)
    p.grad, q.grad = g.clone(), g.clone()
    opt.step()
    opt2.step()
    assert opt2.step_count == 2 and torch.equal(p, q)


def _regression_objective(net, seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    teacher = torch.nn.Linear(8, 2)
    mats, rest = hilo.split_params(model)
    opt, adam = hilo.Hilo(mats, total_steps=10, lr=1e-2, net=net), torch.optim.Adam(rest, lr=1e-3)
    for _ in range(10):
        x = torch.randn(32, 8)
        torch.nn.functional.mse_loss(model(x), teacher(x).detach()).backward()
        opt.step()
        adam.step()
        model.zero_grad(set_to_none=True)
    x = torch.randn(256, 8)
    return torch.nn.functional.mse_loss(model(x), teacher(x)).item()


def _make_regression_objective(device):
    torch.set_num_threads(1)
    return _regression_objective


def test_meta_train_serial_and_parallel():
    """Two ES iterations; worker processes give exactly the serial result."""
    import hilo.meta

    cfg = hilo.HiloNetConfig(d=8, heads=2, mlp_hidden=8, tie=2)
    with tempfile.TemporaryDirectory() as d:
        net = hilo.meta.meta_train(_regression_objective, hilo.HiloNet(cfg), iters=2, pairs=2, out=d)
        assert (Path(d) / "hilo_final.pt").exists() and (Path(d) / "log.jsonl").exists()
    par = hilo.meta.meta_train(net=hilo.HiloNet(cfg), iters=2, pairs=2, make_objective=_make_regression_objective,
                               devices=["cpu", "cpu"])
    assert net.cfg == cfg and torch.isfinite(net.flat()).all()
    assert torch.allclose(net.flat(), par.flat(), atol=1e-6), (net.flat() - par.flat()).abs().max()


def test_truncated_estimators_unbiased():
    """On x <- A x + B theta with loss |x - c|^2 per step, PES and NRES summed over the segments of one run match
    the autograd gradient of the whole run (the smoothed gradient of a quadratic is its gradient)."""
    import hilo.meta

    A, B = torch.tensor([[0.9, 0.2], [-0.1, 0.8]]), torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
    c, theta, T, K = torch.tensor([1.0, -2.0]), torch.tensor([0.3, -0.2, 0.1]), 20, 5

    def segment(x, th, steps):
        loss = 0.0
        for _ in range(steps):
            x = A @ x + B @ th
            loss = loss + (x - c).square().sum()
        return x, loss

    th = theta.clone().requires_grad_()
    segment(torch.zeros(2), th, T)[1].backward()
    sigma = torch.full((3,), 0.1)
    for estimator in hilo.meta.ESTIMATORS:
        states = {}

        def unroll(jobs):
            out = []
            for pid, th, steps, reset in jobs:
                if reset is not None:
                    states[pid] = torch.zeros(2)
                states[pid], loss = segment(states[pid], th, steps)
                out.append(float(loss))
            return out

        est = hilo.meta.TruncatedES(unroll, sigma, pairs=4000, trunc=K, horizon=T, estimator=estimator,
                                    stagger=False, shaping="none")
        ghat = sum(est(theta, it)[0] for it in range(T // K)) / (2 * sigma)
        err = (ghat - th.grad).norm() / th.grad.norm()
        assert err < 0.05, (estimator, ghat, th.grad)


class _RegressionTask:
    def init(self, task_seed, horizon):
        torch.manual_seed(task_seed)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        mats, rest = hilo.split_params(model)
        return dict(model=model, teacher=torch.nn.Linear(8, 2), opt=hilo.Hilo(mats, total_steps=horizon, lr=1e-2),
                    adam=torch.optim.Adam(rest, lr=1e-3), gen=torch.Generator().manual_seed(task_seed))

    def unroll(self, net, st, steps):
        st["opt"].net = net
        total = 0.0
        for _ in range(steps):
            x = torch.randn(32, 8, generator=st["gen"])
            loss = torch.nn.functional.mse_loss(st["model"](x), st["teacher"](x).detach())
            loss.backward()
            total += loss.item()
            st["opt"].step()
            st["adam"].step()
            st["model"].zero_grad(set_to_none=True)
        return st, total / steps


def _make_regression_task(device):
    torch.set_num_threads(1)
    return _RegressionTask()


def test_meta_train_truncated_serial_and_parallel():
    """PES and NRES over runs of random length; worker processes give exactly the serial result."""
    import hilo.meta

    cfg = hilo.HiloNetConfig(d=8, heads=2, mlp_hidden=8, tie=2)
    for estimator, optimizer, shaping in [("pes", "sgdm", "rms"), ("nres", "adam", "rank")]:
        kw = dict(horizon=(4, 9), trunc=3, estimator=estimator, iters=8, pairs=2, optimizer=optimizer,
                  shaping=shaping, alpha=0.1)
        with tempfile.TemporaryDirectory() as d:
            ser = hilo.meta.meta_train_truncated(_RegressionTask(), hilo.HiloNet(cfg), out=d, **kw)
            assert (Path(d) / "hilo_final.pt").exists()
            assert len((Path(d) / "log.jsonl").read_text().splitlines()) == 8
        par = hilo.meta.meta_train_truncated(net=hilo.HiloNet(cfg), make_task=_make_regression_task,
                                             devices=["cpu", "cpu"], **kw)
        assert torch.isfinite(ser.flat()).all() and not torch.equal(ser.flat(), hilo.HiloNet(cfg).flat())
        assert torch.allclose(ser.flat(), par.flat(), atol=1e-6), (estimator, (ser.flat() - par.flat()).abs().max())


def test_shaping():
    import hilo.meta

    L = torch.tensor([[1.0, 3.0], [2.0, 0.0]], dtype=torch.float64)
    assert torch.equal(hilo.meta.shape(L, "none"), torch.tensor([-2.0, 2.0], dtype=torch.float64))
    assert torch.allclose(hilo.meta.shape(L, "rank"), torch.tensor([-2 / 3, 2 / 3], dtype=torch.float64))
    assert torch.allclose(hilo.meta.shape(L, "rms"), torch.tensor([-1.0, 1.0], dtype=torch.float64))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
