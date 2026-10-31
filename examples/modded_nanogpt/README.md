# Using Hilo in modded-nanogpt

The example is the **official baseline** #36 of modded-nanogpt Track 3 (the optimization track),
[`records/track_3_optimization/train_gpt_simple.py`](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/train_gpt_simple.py):
a GPT with 12 layers × 768 dims, 524288 tokens per step, 3250 steps; embeddings / output head / gains and biases use
AdamW, and the hidden matrices originally use Muon.

`train_gpt_hilo.py` is that official script with three changes (all in [`train_gpt_hilo.diff`](train_gpt_hilo.diff)):

1. Muon is removed (Newton–Schulz and the `Muon` class).
2. The hidden matrices go to `Hilo.optimizer`:

   ```python
   import Hilo

   optimizer2 = Hilo.optimizer([p for p in model.blocks.parameters() if p.ndim >= 2], total_steps=train_steps,
                               lr=float(os.environ.get("HILO_LR", 1e-3)), net=os.environ.get("HILO_NET") or None)
   ```

3. The original "stable, then linear decay" learning-rate schedule applies only to AdamW; the lr of
   `Hilo.optimizer` stays constant.

Model, data, init, AdamW and validation are unchanged.

## Running

Run from the root of the modded-nanogpt repository (it needs `data/fineweb10B`; download it with
`python data/cached_fineweb10B.py 9`), after `pip install -e <Hilo repo>`.

```bash
# train with a meta-trained HiloNet (HILO_LR as in meta-training)
HILO_NET=runs/proxy/hilo_final.pt HILO_LR=1e-3 torchrun --standalone --nproc_per_node=8 train_gpt_hilo.py

# without HILO_NET: a randomly initialized HiloNet
torchrun --standalone --nproc_per_node=8 train_gpt_hilo.py

# baseline: run the official train_gpt_simple.py directly
```

As in the original script, the last command-line argument is the number of trials (`num_trials`).

## Meta-training a HiloNet: `meta_train_proxy.py`

ES at full scale is too expensive, so meta-training runs on a scaled-down `train_gpt_hilo.py`: the same architecture,
init, AdamW groups and schedule, loss and constant Hilo lr, with smaller width, depth, batch and number of steps
(default 4 layers × 256 dims, 300 steps, 32k tokens per step, one GPU). Every ES evaluation trains a proxy model
from scratch and returns its final validation loss.

Meta-training starts from a random init, or from a saved HiloNet with `--init net.pt`.

```bash
python meta_train_proxy.py --out runs/proxy --iters 100 --pairs 8                       # from a random init
python meta_train_proxy.py --config '{"output_norm": "rms", "tie": 6}' --out runs/proxy_rms  # other architecture / update norm
python meta_train_proxy.py --init runs/proxy/hilo_final.pt --steps 600 --out runs/proxy2  # continue on a longer proxy
python meta_train_proxy.py --eval runs/proxy2/hilo_final.pt --against runs/proxy/hilo_final.pt --seeds 8
```

`--optimizer`, `--shaping`, `--alpha` and `--momentum` are the arguments of `Hilo.meta.train` (`Hilo/meta`).
Each outer step appends one line to `out/log.jsonl`.
`--eval` evaluates on held-out tasks disjoint from meta-training (final validation loss); with `--against`, two
HiloNets are compared pairwise on the same tasks.

Options related to efficiency:

- `--devices cuda:0,cuda:1,...`: spread the evaluations of one outer step over several worker processes; a GPU may be
  listed several times.
- `--precision`, `--backend`, `--compile-net`: execution options of the HiloNet. Pass the same values when using the saved net.
- `--compile-model`: compile the proxy GPT.
- Evaluations use deterministic CUDA kernels by default, so the same net on the same task gives exactly the same
  result; `--nondeterministic` turns this off.

The closer the proxy is to the target setting, the better the transfer; a common recipe is to train on a small proxy
first, then continue on progressively wider and longer proxies.
