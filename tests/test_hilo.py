"""Run with `python -m pytest tests` or `python tests/test_hilo.py` (CPU, under a minute)."""
import copy
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Hilo  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


def test_hilonet_matches_reference():
    """10 steps of a trained HiloNet (parameters stored in the fixture) against the reference implementation."""
    fx = torch.load(FIX / "hilonet_ref.pt")
    net = Hilo.net(Hilo.config(**fx["config"]), phi=fx["phi"], precision="fp32")
    ps = [w.clone().requires_grad_() for w in fx["W0"]]
    opt = Hilo.optimizer(ps, total_steps=fx["T"], lr=fx["lr"], net=net)
    for gs in fx["grads"]:
        for p, g in zip(ps, gs):
            p.grad = g.clone()
        opt.step()
    for p, w in zip(ps, fx["W_final"]):
        assert torch.allclose(p.detach(), w, atol=1e-5), (p.detach() - w).abs().max()


def test_triton_backend_matches_reference():
    """On CUDA with Triton: the fused bf16 kernels agree with the float32 reference (skipped otherwise)."""
    import importlib.util

    if importlib.util.find_spec("triton") is None or not torch.cuda.is_available():
        return
    for attn, tie in (("linear", 3), ("softmax", 0)):
        cfg = Hilo.config(d=16, heads=2, mlp_hidden=32, tie=tie, attn=attn)
        flat = Hilo.net(cfg).flat()
        flat = flat + 0.05 * torch.randn(flat.numel(), generator=torch.Generator().manual_seed(0))
        fast = Hilo.net(cfg, phi=flat, precision="bf16", backend="triton")
        ref = Hilo.net(cfg, phi=flat, precision="fp32", backend="torch")
        g = torch.Generator().manual_seed(1)
        for m, n in ((40, 24), (96, 160)):
            G, W = torch.randn(3, m, n, generator=g).cuda(), 0.02 * torch.randn(3, m, n, generator=g).cuda()
            M = torch.randn(3, m, n, len(cfg.betas), generator=g).cuda()
            a, b = fast(G, M, W, 0.4), ref(G, M, W, 0.4)
            assert fast._triton is not None and ((a - b).norm() / b.norm()).item() < 1e-2, (attn, m, n)


def test_hilonet_random_init_save_load():
    cfg = Hilo.config(d=8, heads=2, mlp_hidden=16, tie=3, attn="softmax", betas=(0.9, 0.99), output_norm="rms")
    a, b = Hilo.net(cfg, seed=1), Hilo.net(cfg, seed=2)
    assert not torch.equal(a.flat(), b.flat()) and torch.equal(a.flat(), Hilo.net(cfg, seed=1).flat())
    with tempfile.TemporaryDirectory() as d:
        a.save(Path(d) / "net.pt")
        c = Hilo.optimizer([torch.zeros(2, 2)], total_steps=1, net=str(Path(d) / "net.pt")).net
    assert c.cfg == cfg and torch.equal(c.flat(), a.flat())
    p = torch.randn(12, 20, requires_grad=True)
    p.grad = torch.randn(12, 20)
    Hilo.optimizer([p], total_steps=5, lr=0.1, net=c).step()
    assert torch.isfinite(p).all()


def test_output_norms():
    torch.manual_seed(1)
    for mode in Hilo.OUTPUT_NORMS:
        net = Hilo.net(Hilo.config(output_norm=mode))
        for shape in [(64, 16), (16, 64), (32, 32)]:
            p = torch.zeros(shape, requires_grad=True)
            p.grad = torch.randn(shape)
            Hilo.optimizer([p], total_steps=10, lr=1.0, net=net).step()
            norm = p.detach().norm().item()
            if mode == "none":
                assert 0 < norm < float("inf")
            else:
                expect = Hilo.norms.target_norm(mode, *shape)
                assert abs(norm - expect) < 1e-3 * expect, (mode, shape, norm, expect)
    x = torch.randn(3, 8, 4)
    assert torch.equal(Hilo.normalize(x, "none"), x)
    assert torch.allclose(Hilo.normalize(x, "rms").square().mean(dim=(1, 2)), torch.ones(3))


def test_weight_decay():
    """Decoupled: the run with weight decay differs from the one without by -lr * wd * W0."""
    torch.manual_seed(4)
    w0, g = torch.randn(6, 4), torch.randn(6, 4)
    out = []
    for wd in (0.0, 0.5):
        p = w0.clone().requires_grad_()
        p.grad = g.clone()
        Hilo.optimizer([p], total_steps=10, lr=0.1, weight_decay=wd).step()
        out.append(p.detach())
    assert torch.allclose(out[1] - out[0], -0.1 * 0.5 * w0, atol=1e-6)


def test_assign_owners_balances_cost():
    from Hilo.optim import assign_owners
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
    opt = Hilo.optimizer([p], lr=0.1, total_steps=10)
    p.grad = torch.randn(8, 4)
    opt.step()
    q = p.detach().clone().requires_grad_()
    opt2 = Hilo.optimizer([q], lr=0.1, total_steps=10)
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
    mats, rest = Hilo.split_params(model)
    opt, adam = Hilo.optimizer(mats, total_steps=10, lr=1e-2, net=net), torch.optim.Adam(rest, lr=1e-3)
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


def test_meta_es_serial_and_parallel():
    """Two ES iterations per outer optimizer; worker processes give exactly the serial result."""
    import Hilo.meta

    cfg = Hilo.config(d=8, heads=2, mlp_hidden=8, tie=2)
    for optimizer, shaping, alpha in [("sgdm", "rms", 1.0), ("adam", "rank", 0.1)]:
        kw = dict(iters=2, pairs=2, optimizer=optimizer, shaping=shaping, alpha=alpha)
        with tempfile.TemporaryDirectory() as d:
            net = Hilo.meta.train(_regression_objective, Hilo.net(cfg), out=d, **kw)
            assert (Path(d) / "hilo_final.pt").exists() and (Path(d) / "log.jsonl").exists()
        par = Hilo.meta.train(net=Hilo.net(cfg), make_task=_make_regression_objective, devices=["cpu", "cpu"], **kw)
        assert net.cfg == cfg and torch.isfinite(net.flat()).all()
        assert not torch.equal(net.flat(), Hilo.net(cfg).flat())
        assert torch.allclose(net.flat(), par.flat(), atol=1e-6), (optimizer, (net.flat() - par.flat()).abs().max())


def test_shaping():
    import Hilo.meta

    L = torch.tensor([[1.0, 3.0], [2.0, 0.0]], dtype=torch.float64)
    assert torch.equal(Hilo.meta.shape(L, "none"), torch.tensor([-2.0, 2.0], dtype=torch.float64))
    assert torch.allclose(Hilo.meta.shape(L, "rank"), torch.tensor([-2 / 3, 2 / 3], dtype=torch.float64))
    assert torch.allclose(Hilo.meta.shape(L, "rms"), torch.tensor([-1.0, 1.0], dtype=torch.float64))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
