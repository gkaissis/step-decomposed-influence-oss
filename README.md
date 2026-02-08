# Step-Decomposed Influence (SDI)

Step-Decomposed Influence (SDI) decomposes TracIn into a **step-resolved influence trajectory** for looped/weight-tied models. This reference implementation computes **per-step influence** while avoiding per-example gradient materialization by sketching **during** backprop (TensorSketch for 2D weights, CountSketch for 1D params).

Paper (placeholder): https://arxiv.org/abs/XXXX.XXXXX

## Install (UV-first)

```bash
# From PyPI
uv add step-decomposed-influence

# From source (editable, in a UV project)
uv add --editable .
```

CPU vs CUDA:
- This package is device-agnostic. CUDA works automatically if you install a CUDA-enabled PyTorch.
- Follow the official PyTorch install instructions to choose the right wheel for your system.

## Quickstart

```python
import torch
import torch.nn.functional as F
from sdi import ProjectedTracInSDI, CheckpointSpec

# model = your looped transformer
# target_modules = the recurrent core used at each loop step

def loss_fn(model, batch):
    logits = model(batch["tokens"])  # (B, T, vocab)
    targets = batch["tokens"][:, 1:]
    logits = logits[:, :-1, :]
    per_pos = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.size(0), -1)
    return per_pos.mean(dim=1)  # per-example loss (B,)

est = ProjectedTracInSDI(
    model=model,
    target_modules=target_modules,
    projection_size=2048,
    loss_reduction="sum",
)

out = est.influence_across_checkpoints(
    checkpoints=[CheckpointSpec("checkpoints/ckpt_0001.pt")],
    train_loader=train_loader,
    query_loader=query_loader,
    loss_fn=loss_fn,
    mode="sdi",
    train_chunk_size=128,
    query_chunk_size=128,
)

# out.sdi:    (N_train, N_query, steps_query)
# out.tracin: (N_train, N_query)
```

## Example

Run the toy looped-transformer demo:

```bash
uv run python examples/toy_looped_transformer_sdi.py --device cuda
```

This script:
- trains a tiny looped transformer on random data,
- saves a few checkpoints,
- computes **projected SDI** (sketch-during-backprop),
- and shows **fast TracIn** (scalar-only) via `mode="tracin"`.

## API (minimal)

- `ProjectedTracInSDI`: projected SDI/TracIn using TensorSketch + CountSketch.
- `FullGradientTracInSDI`: exact baseline (materializes per-sample gradients; small models only).
- `CheckpointSpec`: path + optional checkpoint weight (eta).
- `InfluenceOutputs`: `sdi` (None when `mode="tracin"`), `tracin`, and optional `sdi_matrix`.

## Assumptions & limitations

- **Steps** are inferred from how often the selected parameters are used in a single forward/backward. For looped transformers, this equals the loop horizon when you pass only the looped core.
- **Projected estimator supports**:
  - 2D params: `nn.Linear.weight`, `nn.Embedding.weight` (TensorSketch)
  - 1D params: CountSketch of per-sample grads via Opacus grad samplers (plus analytic `nn.Linear.bias`)
- **Custom modules** without Opacus grad-sampler coverage are not supported in strict mode.
- **Checkpoint loading** uses `torch.load`. Do not load untrusted checkpoints.

## Citation

For citing this software repository, use GitHub's "Cite this repository"
button.

For citing the paper, use:

```bibtex
@article{kaissis2026sdi,
  title  = {Step-Resolved Data Attribution for Looped Transformers},
  author = {Georgios Kaissis and David Mildenberger and Felipe Gomez and Martin Menten and Eleni Triantafillou},
  year   = {2026},
  note   = {arXiv preprint},
}
```

## Authors

- Georgios Kaissis (Hasso-Plattner Institute)
- David Mildenberger (Technical University of Munich)
- Felipe Gomez (Harvard University)
- Martin Menten (Technical University of Munich)
- Eleni Triantafillou (Google DeepMind)
