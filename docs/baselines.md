# Baselines: Published Split-Learning Results, Reproduced

`baselines/` holds published SL results that rerun in TorchSlicer from one command, each
with the paper's number and a tolerance. A baseline is a claim: *this setting, in
TorchSlicer, gives the paper's number*. It either reproduces the number, or says plainly
that it doesn't and why.

```bash
.venv/bin/python baselines/run.py slperf_mnist --smoke        # a minute: does it run?
.venv/bin/python baselines/run.py slperf_mnist                # all runs, all seeds
.venv/bin/python baselines/run.py slperf_mnist --seed 0 &     # or seeds in parallel...
.venv/bin/python baselines/run.py slperf_mnist --seed 1 &
.venv/bin/python baselines/run.py slperf_mnist --merge        # ...then merge
.venv/bin/python baselines/run.py slperf_mnist --check        # exit 1 if outside tolerance
```

## What a baseline is

`baselines/<name>/`:

- `config.yaml`: a TorchSlicer `RunConfig` (the setting the runs share: `training`, `data`,
  `sl`, ...) plus a `baseline` section, which TorchSlicer itself ignores:
  - `paper`: title, link, and the table or figure the numbers come from;
  - `model`: a builder in `baselines/models.py` and its arguments;
  - `metric` (a history key, e.g. `test_accuracy`) and `report` (`last` or `best` epoch:
    whichever the paper reports);
  - `seeds` (at least 3), `smoke` (a tiny override), `synthetic` (input shape and classes,
    for the CI test; or `{data, model, cuts}` overrides, for a baseline that needs its own
    data or a smaller model in CI);
  - `runs`: each a `label`, its `cuts`, a `config` override and `expected: {value, tol}`.
- `README.md`: the paper, the setting, a table of published vs ours (mean ± std), the
  tolerance, and every deviation from the paper with its reason.
- `results/<name>.json`: the merged result (per-seed values, curves, communication bytes).

## Adding one

1. Pick a paper whose setting you can state completely: model, cut, data split, clients,
   optimizer, epochs, what is reported. Read the paper's text and, if there is one, its code.
   Where they disagree, follow the one that produced the number (usually the code), and
   say so in the README.
2. Add the model to `baselines/models.py` as a flat `nn.Sequential`, so cuts are layer
   indices. What only one baseline needs goes in its `extras.py`, which the runner imports
   first: a dataset (`ts.data.register_dataset`), a partitioner, a model builder (added to
   `models.MODELS`), and an `evaluate(model, cfg, run, smoke) -> {metric: value}` that runs
   after training when the metric is not in the training history (e.g. generation scores).
3. Write `config.yaml`. One run per number you reproduce. Choose the tolerance before
   running: about two standard deviations of what the paper's setting gives between seeds,
   wider for non-IID runs, which vary a lot.
4. `--smoke`, then the full run, then `--check`.
5. Write the README. "Not reproduced" is a valid result: give the gap and what you know
   about why (an unstated hyperparameter, a bug in the reference code, a setting
   TorchSlicer cannot express yet).

`tests/test_baselines.py` runs every baseline's runs for one epoch on synthetic data, so a
broken config fails CI. It does not check the numbers: that takes the real data and time.
