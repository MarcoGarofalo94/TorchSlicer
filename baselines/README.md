# Baselines

Published split-learning results, rerun in TorchSlicer. How they work and how to add one:
[docs/baselines.md](../docs/baselines.md).

| baseline | paper | what | status |
|---|---|---|---|
| [slperf_mnist](slperf_mnist/) | SLPerf (Zhou et al., 2023), Table 2 | vanilla SL, U-shaped SL, SplitFed; MNIST LeNet, 3 clients, IID and Dirichlet(0.1) | not reproduced (the authors' code does not reproduce it either; TorchSlicer matches that code on the same split) |
| [label_leakage](label_leakage/) | Li et al. (ICLR 2022), §5 | norm and direction label attacks on the cut gradients, no noise and ClipNoise; CIFAR-10 binary stand-in | reproduced (AUC ≈ 1 without noise) |
| [splitfed_fmnist](splitfed_fmnist/) | SplitFed (Thapa et al., AAAI 2022), Table 5 | SL, SFL-V1, SFL-V2; Fashion-MNIST LeNet, 5 clients | reproduced (all within 0.01) |
| [splitlora_e2e](splitlora_e2e/) | SplitLoRA (Lin et al., 2024), Table I | CenLoRA vs SplitLoRA; GPT-2 small, LoRA r 4, E2E NLG, 3 clients | not reproduced (12.5 BLEU below in both settings, 1 seed); split = centralized as in the paper |

```bash
.venv/bin/python baselines/run.py <name> [--smoke | --seed k | --merge | --check]
```
