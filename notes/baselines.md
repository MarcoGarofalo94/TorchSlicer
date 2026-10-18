# M13 baselines + Flower interop: plan (2026-10-18)

Goal for adoption: a published SL result reruns in TorchSlicer from one command, with the
expected number and a tolerance checked, like Flower Baselines.

## Template (`baselines/<name>/`, shipped)
- `README.md`: paper (title, venue, link), the setting (model, cut, data split, clients,
  optimizer, epochs), the published number(s), our number (mean ± std, ≥ 3 seeds), tolerance,
  and every deviation from the paper with its reason.
- `config.yaml`: a RunConfig plus a `baseline:` section (runs, seeds, expected
  `{metric, value, tol}` per run). One shared runner `baselines/run.py <name> [--smoke |
  --seed k | --merge | --check]`; `--check` fails if a merged result is outside tolerance.
- `results/`: merged JSON. CI: `tests/test_baselines.py` runs every baseline's `--smoke`
  (tiny subset, 1 epoch: it runs and reports the metrics, not the number).
- `docs/baselines.md`: contributor guide (cf. Flower Baselines): how to add one, what the
  README must state, how tolerance is chosen, "not reproduced" is a valid outcome.

## Candidates (sources read 2026-10-18; numbers from the papers' tables)
1. **SLPerf** (Zhou et al. 2023, https://arxiv.org/abs/2304.01502, Table 2; code
   https://github.com/Rainysponge/Split-learning-Attacks): MNIST, LeNet (6c5-pool-16c5-pool-
   120-84-10, ReLU), split after the first conv block, 3 clients, batch 64, lr 0.01; test acc
   IID / Dirichlet α = 0.1: SplitFed 0.9882 / 0.9768, Vanilla 0.9792 / 0.7288, U-shape
   0.9728 / 0.7019 (centralized 0.9843). Covers SplitNN (vanilla), U-shaped, SplitFed.
   Open points: the paper says SGD; the code has SGD momentum 0.9 wd 5e-4 for vanilla/U-shape
   and Adam for SplitFed `[code read]`; epochs: Fig. 6 x-axis 0–30 for MNIST `[unverified:
   number of epochs for Table 2]`; SplitFed version (V1/V2) to check in the code; the
   Dirichlet partitioner's exact form to check (ours: `ts.data.partition(..., "dirichlet")`).
2. **SplitFed** (Thapa et al., AAAI 2022, https://arxiv.org/abs/2004.12088, Table 5): FMNIST
   LeNet, 5 clients, 200 global epochs, batch 1024, lr 0.004 (optimizer not stated in the text
   `[unverified]`), best test acc over 200 epochs: SL 90.4 %, SFL-V1 89.6 %, SFL-V2 90.4 %
   (normal 92.7 %). The official repo has only the ResNet18/HAM10000 scripts (HAM10000 is not
   available here); their LeNet definition is `[unverified]`, use SLPerf's.
3. **Attack + defence**: Li et al. ICLR 2022 label leakage (no noise: leak AUC ≈ 1; from M12),
   with ClipNoise as the defence (our curve; the paper's Marvell is not implemented).
4. **Local loss / decoupled SL**: the paper `LocalLossAlgorithm` cites (Zihad et al.,
   https://arxiv.org/abs/2601.19261) uses ResNet-110 on CIFAR-10/100: hours per run here →
   needs an owner decision or a cheaper paper.
5. **Compression**: no paper with a cheap published number in sota.md yet: search needed.
6. **Split LLM / LoRA**: SplitLoRA (https://arxiv.org/abs/2407.00952): read its setting first.
7. **Flower interop**: `flwr` 1.40 is installable (optional extra): a TorchSlicer client as a
   Flower `NumPyClient` (client slices' weights as its parameters), FedAvg from Flower as the
   SplitFed aggregation. Check against `SplitFedV1Algorithm` on the same data.

Order: template + SLPerf (1) → SplitFed (2) → attack+defence (3) → Flower (7) → (6), (5), (4).
Runs > 30 min: ask the owner first (parallel seeds as processes keep most under that).
