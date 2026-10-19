# SLPerf, Table 2: MNIST, three SL paradigms

**Paper.** Zhou, Hu, Wu, Chen, *SLPerf: a Unified Framework for Benchmarking Split Learning*,
2023, https://arxiv.org/abs/2304.01502 (Table 2, 3 clients). Code: SLFrame in
https://github.com/Rainysponge/Split-learning-Attacks.

**Status: not reproduced.** No run lands inside the tolerance. The IID runs come within
0.011–0.017 of the paper. The non-IID runs differ a lot, and the ranking is reversed (the
paper: SplitFed > vanilla; ours: vanilla > SplitFed). Running the authors' code (SLFrame) does
not give the published vanilla number either, and its U-shaped variant is a different
algorithm (see below).

## Setting

| | |
|---|---|
| model | LeNet: 6c5-ReLU-pool-16c5-ReLU-pool-120-84-10 (`baselines/models.py: lenet`) |
| cut | after the first conv block (`cuts: [2]`); U-shaped adds a cut before the output layer (`[2, 6]`) |
| data | MNIST, 3 clients; IID, or SLPerf's Dirichlet(0.1) "hetero" split (`extras.py`) |
| clients | sequential; vanilla/U-shaped hand the client model to the next client; SplitFed keeps one client model per client, FedAvg (sample-weighted) after each epoch, one server model (V2) |
| optimizer | vanilla/U-shaped: SGD lr 0.01, momentum 0.9, weight decay 5e-4; SplitFed: Adam lr 0.01 (both from the code) |
| batch, epochs | 64, 30 |
| metric | test accuracy on the full MNIST test set after the last epoch (`report: last`) |

## Results (3 seeds, mean ± std)

| run | ours | paper | tol | ok |
|---|---|---|---|---|
| vanilla, IID | 0.9902 ± 0.0012 | 0.9792 | 0.01 | no |
| vanilla, Dir(0.1) | 0.8982 ± 0.0545 | 0.7288 | 0.10 | no |
| U-shaped, IID | 0.9899 ± 0.0008 | 0.9728 | 0.01 | no |
| U-shaped, Dir(0.1) | 0.9385 ± 0.0046 | 0.7019 | 0.10 | no |
| SplitFed, IID | 0.9755 ± 0.0031 | 0.9882 | 0.01 | no |
| SplitFed, Dir(0.1) | 0.8021 ± 0.0231 | 0.9768 | 0.02 | no |

The paper does not say which epoch Table 2 reports. With the best epoch instead of the last,
SplitFed IID is 0.9853 ± 0.0012 (inside the tolerance) and SplitFed Dir(0.1) is 0.854. The
other runs change by ≤ 0.02 and stay outside. We report the last epoch, as configured before the runs. Wall time per run: 5–12 min on
one GPU shared by 9 processes. Raw numbers: `results/slperf_mnist.json`.

## The authors' code (SLFrame), run here

The same setting, seed 0, 30 epochs, CPU, last epoch. SLFrame's split is fixed: it is always
NumPy seed 0, set as a side effect of importing its CheXpert module. The value shown is the last
client's accuracy; for U-shaped it is the mean of the three clients' own models.

| run | paper | SLFrame | TorchSlicer (3 seeds) |
|---|---|---|---|
| vanilla, IID | 0.9792 | 0.987 | 0.9902 ± 0.0012 |
| vanilla, Dir(0.1) | 0.7288 | 0.946 (0.959 with its optimizer bug fixed) | 0.8982 ± 0.0545 |
| U-shaped, IID | 0.9728 | 0.988 | 0.9899 ± 0.0008 |
| U-shaped, Dir(0.1) | 0.7019 | 0.738 (clients 0.649 / 0.767 / 0.799) | 0.9385 ± 0.0046 |
| SplitFed, IID | 0.9882 | 0.980 | 0.9755 ± 0.0031 |
| SplitFed, Dir(0.1) | 0.9768 | 0.932 (0.60–0.94 over epochs) | 0.8021 ± 0.0231 |

- SLFrame does not reproduce Table 2 either. Its vanilla Dir(0.1) is 0.946 at 30 epochs; it is at
  0.71–0.73 after 2 epochs, which is the published value.
- U-shaped Dir(0.1): SLFrame's U-shaped is a different algorithm (below), and that algorithm
  gives roughly the published 0.70.
- **Still open:** SplitFed Dir(0.1), 0.93 in SLFrame against 0.80 here. The splits differ (SLFrame
  always uses its seed-0 draw), and at α = 0.1 with 3 clients the split varies a lot between
  seeds (ours: 0.78–0.82). Training TorchSlicer on SLFrame's exact split would separate the
  split from the implementation; that has not been done yet.

How SLFrame was run: an MPICH venv, and a driver that adds compatibility shims for current
torch/NumPy, stubs two unused imports, caps SLFrame's message-polling sleeps (speed only: a
1-epoch run gives identical accuracies with and without the caps) and caps the U-shaped loop
at 30 rounds.

## Deviations and what we found in the reference code

- **Optimizers.** The paper's text says SGD for every paradigm. SLFrame uses SGD with momentum
  and weight decay for vanilla/U-shaped and Adam for SplitFed. We follow the code.
- **Non-IID split.** `ts.data`'s `dirichlet` is not SLPerf's split. `extras.py` registers
  `slperf_hetero`, a port of SLFrame's `hetero` partition. It draws Dirichlet shares per class,
  redraws until every client has ≥ 10 samples, then truncates every client to the smallest
  one's size.
- **SLFrame vanilla handoff bug (confirmed by running).** At each handoff
  `vanilla/client_manager.py` rebinds the client model (`self.trainer.model = torch.load(...)`),
  but the client's optimizer, created once in `vanilla/client.py`, still holds the old
  parameters. After the first handoff the client layers stop training. Rebuilding the
  optimizer after each load gives 0.959 instead of 0.946 (+1.3 points): the bug is real but
  small. TorchSlicer trains the client layers the whole time.
- **SLFrame U-shaped is a different algorithm.** Its clients never hand their head and tail
  on: each client trains private client parts, and only the server part is shared. Under
  Dir(0.1) each client's tail is biased toward its own classes: 0.738 on average after 30
  rounds, close to the published 0.70. TorchSlicer's U-shaped hands the client parts on like vanilla.
  SLFrame's U-shaped loop also never terminates: it counts each round twice and stops at
  `== 3`.
- **Evaluation.** Each SLFrame client evaluates on the full MNIST test set after its own
  turn. Our evaluation after the last client of each epoch is the same protocol for the last
  client.
- **Not explained.** SplitFed under Dir(0.1): 0.80 here, 0.93 in SLFrame, 0.98 in the paper.
  We found no algorithmic difference in the code we read (FedAvg every epoch over the client
  models, one server model, Adam on both sides). See "Still open" above.

```bash
.venv/bin/python baselines/run.py slperf_mnist --seed 0 --runs vanilla_iid   # one run, one seed
.venv/bin/python baselines/run.py slperf_mnist --merge && .venv/bin/python baselines/run.py slperf_mnist --check
```
