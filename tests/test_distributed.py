"""Sharded Hilo (2 CPU processes, gloo) gives the same weights as one process. `python tests/test_distributed.py`"""
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hilo  # noqa: E402

SHAPES = [(24, 16), (16, 24), (20, 20), (24, 16), (12, 12)]


def run(steps=5):
    torch.manual_seed(0)
    ps = [(torch.randn(s) * 0.1).requires_grad_() for s in SHAPES]
    opt = hilo.Hilo(ps, lr=0.02, total_steps=steps)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()
    return [p.detach() for p in ps]


def worker(rank, world, out):
    os.environ |= {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29533"}
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.save(run(), f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


def test_sharded_matches_single():
    ref = run()
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(worker, args=(2, d), nprocs=2)
        for r in range(2):
            for a, b in zip(torch.load(f"{d}/rank{r}.pt"), ref):
                assert torch.allclose(a, b, atol=1e-6), (r, (a - b).abs().max())


if __name__ == "__main__":
    test_sharded_matches_single()
    print("ok test_sharded_matches_single")
