# SplitLoRA, Table I: GPT-2 small, LoRA r = 4, E2E NLG

**Paper.** Lin et al., *SplitLoRA: A Split Parameter-Efficient Fine-Tuning Framework for Large
Language Models*, 2024, https://arxiv.org/abs/2407.00952 (Table I, GPT2-S, r = 4). No public code.

**Status: not reproduced (1 seed).** Both settings score about 12.5 BLEU below the paper. The
paper's main claim does hold: split training matches centralized training, with a gap of
0.6 BLEU in the paper and none here.

## Setting

| | |
|---|---|
| model | GPT-2 small (Hugging Face `gpt2`), frozen; LoRA on q and v of every block, r 4, alpha 32, LoRA dropout 0.1, no other dropout |
| cut | client: embeddings + the first 3 blocks; server: the other 9 blocks + the head |
| CenLoRA | one client, same cut |
| SplitLoRA | 3 clients, IID, one at a time, SplitFed aggregation (FedAvg of the client adapters) every epoch |
| training | AdamW lr 2e-4, weight decay 0.01, batch 8, 5 epochs, 500 warmup steps then linear decay, label smoothing 0.1, loss on the reference tokens only |
| decoding | beam 10, length penalty 0.8, no repeated 4-grams, at most 64 new tokens |
| metric | the official E2E scorer (`measure_scores.py`, tuetschek/e2e-metrics), all references per input |

The paper gives the learning rate, batch, rank, cut and client count. Everything else
(epochs, optimizer, schedule, LoRA targets, decoding) follows the LoRA paper's GPT-2 E2E recipe
(https://github.com/microsoft/LoRA, `examples/NLG`). The data are that repository's E2E files.

## Results (seed 0)

| run | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| CenLoRA, ours | 0.5658 | 7.826 | 0.402 | 0.628 | 1.462 |
| CenLoRA, paper | 0.6941 | 8.782 | 0.461 | 0.707 | 2.471 |
| SplitLoRA, ours | 0.5665 | 7.869 | 0.417 | 0.631 | 1.425 |
| SplitLoRA, paper | 0.6879 | 8.726 | 0.457 | 0.698 | 2.441 |

Wall time: 2 h 15 min per run on one A100 (two runs sharing it). Activation traffic over the
cut: 15.0 GB in total over the 5 epochs. TorchSlicer's per-step time is within 1.5% of the same
model trained in plain PyTorch.

## What was checked

- The generated texts are fluent, mention every input attribute, and have the right length
  (29 words on average against 27 for the references).
- Encoding, loss mask, label smoothing, how outputs are cut off and how references are grouped
  all match the LoRA repository's code.
- Turning off Hugging Face GPT-2's default 0.1 dropout, which the LoRA recipe's GPT-2 does not
  have, changed BLEU by 0.005.

Still unknown: whether the paper used a different recipe. The LoRA paper reports only GPT-2
medium and large. Not run: microsoft/LoRA's own code with GPT-2 small.

## Run

```bash
.venv/bin/python baselines/splitlora_e2e/prepare.py     # data + scorer (needs Java 8+ for METEOR)
.venv/bin/python baselines/run.py splitlora_e2e --seed 0
```
