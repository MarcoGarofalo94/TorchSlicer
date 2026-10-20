# Label leakage from cut gradients (Li et al., ICLR 2022)

**Paper.** Li, Sun, Yang, Gao, Zhang, Xie, Smith, Wang, *Label Leakage and Protection
in Two-party Split Learning*, ICLR 2022, https://arxiv.org/abs/2102.08504.

**Status: reproduced** (the qualitative result: with no protection the leak AUC is ≈ 1).

## Setting

| | |
|---|---|
| task | imbalanced binary classification. The paper uses Criteo, Avazu and ISIC, which are not available here. CIFAR-10 "airplane or not" (10 % positives) stands in; `prepare.py` writes it |
| model | small CNN (`baselines/models.py: cnn`), cut after the first conv block (`cuts: [2]`), vanilla SL: the client holds the inputs and the server holds the labels |
| training | Adam lr 1e-3, batch 64, 3 epochs, 20 000 training samples |
| attacks | norm and cosine-direction scoring of the gradient the server returns (`label_norm`, `label_direction`) |
| metric | leak AUC per batch, summarized by its 95 % quantile over the batches of the last epoch, as in the paper |

## Results (3 seeds, mean ± std)

| run | ours | paper | tol | ok |
|---|---|---|---|---|
| direction attack, no noise | 1.000 ± 0.000 | ≈ 1 | 0.02 | yes |
| norm attack, no noise | 0.993 ± 0.009 | ≈ 1 | 0.05 | yes |
| direction attack, ClipNoise C = 1e-2, z = 1 | 0.746 ± 0.035 | (not in the paper) | - | not checked |

The defence row is our own curve point, not the paper's: per-sample clipping and Gaussian noise
on the returned gradient (`ts.ClipNoise`). It costs ε = 20.1 at δ = 1e-5 over the 3 epochs,
by Gaussian-DP accounting (the `dp` attack). That bound covers each sample's own gradient
row only; it is not end-to-end DP. The 95 % quantile reflects the worst batches. In `benchmarks/privacy` (same data, model, cut
and budget) the mean AUC over batches at z = 1 is 0.50. The paper's own defence (Marvell) is not
implemented. Wall time: under 1 min per seed on one GPU. Raw numbers:
`results/label_leakage.json`.

```bash
.venv/bin/python baselines/label_leakage/prepare.py
.venv/bin/python baselines/run.py label_leakage --seed 0
.venv/bin/python baselines/run.py label_leakage --merge && .venv/bin/python baselines/run.py label_leakage --check
```
