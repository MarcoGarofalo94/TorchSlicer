# SplitFed, Table 5: Fashion-MNIST LeNet

**Paper.** Thapa et al., *SplitFed: When Federated Learning Meets Split
Learning*, AAAI 2022, https://arxiv.org/abs/2004.12088 (Table 5). Code:
https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning.

**Status: reproduced.** All three runs fall inside ±0.01 of the paper.

## Setting

| | |
|---|---|
| model | LeNet (`baselines/models.py: lenet`; the authors' LeNet is not public, so this uses SLPerf's), cut after the first conv block |
| data | Fashion-MNIST, 5 clients, IID, equal size |
| SL | one client at a time; the client model is handed to the next client |
| SFL-V1 | each client trains its own client model and its own copy of the server model; both are FedAvg'd after each epoch |
| SFL-V2 | each client keeps its own client model, there is one server model, clients run in turn; client models are FedAvg'd after each epoch |
| training | Adam lr 0.004 (Adam is from the authors' code; the text gives no optimizer), batch 1024, 200 epochs |
| metric | best test accuracy over the 200 epochs, as in the paper |

## Results (3 seeds, mean ± std)

| run | ours | paper | tol | ok |
|---|---|---|---|---|
| SL | 0.9014 ± 0.0019 | 0.904 | 0.01 | yes |
| SFL-V1 | 0.9012 ± 0.0032 | 0.896 | 0.01 | yes |
| SFL-V2 | 0.8972 ± 0.0039 | 0.904 | 0.01 | yes |

Last-epoch accuracy is 0.4–1.5 points lower: SL 0.890, SFL-V1 0.893, SFL-V2 0.884.
Wall time: 15–20 min per run on one GPU shared by 9 processes. Raw numbers:
`results/splitfed_fmnist.json`.

## Deviations

- LeNet definition: SLPerf's (6c5-pool-16c5-pool-120-84-10, ReLU); the paper does not give its own.
- Client order: fixed, 0..4, every epoch. The authors' ResNet scripts draw a random order;
  with IID equal clients this should not matter.
- FedAvg weights clients by sample count, as in the paper's Algorithm 1. The authors' scripts
  average with equal weights, which is the same thing here because the clients are equal.

```bash
.venv/bin/python baselines/run.py splitfed_fmnist --seed 0 --runs sl
.venv/bin/python baselines/run.py splitfed_fmnist --merge && .venv/bin/python baselines/run.py splitfed_fmnist --check
```
