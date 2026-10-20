# C3-SL, Table 1: VGG-16 on CIFAR-10, batch-wise compression at R = 2, 4, 8, 16

**Paper.** Hsieh, Chuang, Wu, *C3-SL: Circular Convolution-Based Batch-Wise Compression for
Communication-Efficient Split Learning*, MLSP 2022, https://arxiv.org/abs/2207.12397 (Table 1,
VGG-16, CIFAR-10). Code: https://github.com/WesleyHsieh0806/C3-SL.

**Status: reproduced (3 seeds).** Every run is within 0.01 of the paper. The paper's claim
holds: compressing the cut 16× in both directions costs at most 0.3 points of accuracy.

## Setting

| | |
|---|---|
| model | VGG-16, ImageNet-pretrained (torchvision), new 10-class head |
| cut | client: `features` up to the 4th max-pooling layer (512 × 2 × 2 activation); server: the rest |
| compression | `ts.CircularConv(ratio=R)`: each group of R samples is sent as one vector, the sum of the samples circularly convolved with R fixed random keys, and decoded by circular correlation. The gradient goes back compressed the same way |
| training | vanilla SL, one client; Adam lr 1e-4, batch 64, 100 epochs, random horizontal flip |
| evaluation | the CIFAR-10 test set, through the same compression, after every epoch |
| metric | best test accuracy over the 100 epochs, as the authors' code reports it |

All of it follows the paper and the authors' code. `CircularConv` matches the code's
encoder and decoder to within 2e-5.

## Results (mean ± std over seeds 0, 1, 2)

| run | sent per epoch, each direction | best, ours | best, paper | last epoch, ours |
|---|---|---|---|---|
| vanilla SL | 409.6 MB | 0.9045 ± 0.0026 | 0.899 | 0.8955 ± 0.0039 |
| C3-SL, R = 2 | 204.8 MB | 0.9032 ± 0.0014 | 0.903 | 0.8964 ± 0.0039 |
| C3-SL, R = 4 | 102.4 MB | 0.9030 ± 0.0010 | 0.900 | 0.8969 ± 0.0009 |
| C3-SL, R = 8 | 51.2 MB | 0.9019 ± 0.0009 | 0.899 | 0.8917 ± 0.0034 |
| C3-SL, R = 16 | 25.6 MB | 0.9012 ± 0.0014 | 0.896 | 0.8963 ± 0.0052 |

The sizes count the forward activations of the 50,000 training images. The gradients going
back are the same size. Wall time: about 1 h 35 min per run on one A100, with four runs sharing it.

**Deviations from the paper:**
- The paper averages 5 trials; these results average 3 seeds.
- CIFAR-10 comes from local `.npz` copies of the torchvision files, with the same images and
  the same normalization as the authors' code.
- Picking the best epoch on the test set is optimistic, so the last-epoch accuracy is listed
  too. It tells the same story: no run loses more than 0.4 points to vanilla SL.

## Run

```bash
.venv/bin/python baselines/run.py c3sl_cifar10 --seed 0      # each run, ~1.5 h on an A100
.venv/bin/python baselines/run.py c3sl_cifar10 --merge
```
