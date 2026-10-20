"""Write the binary task: CIFAR-10 "is it an airplane?" (10 % positives), from the local
CIFAR-10 npz files, as data/cifar10_airplane_{train,test}.npz.

    .venv/bin/python baselines/label_leakage/prepare.py [--root data]
"""
import os
import sys

import numpy as np

root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "data"
for split in ("train", "test"):
    z = np.load(os.path.join(root, f"cifar10_{split}.npz"))
    np.savez(os.path.join(root, f"cifar10_airplane_{split}.npz"), x=z["x"], y=(z["y"] == 0).astype(np.int64))
    print(split, z["x"].shape, f"positives {(z['y'] == 0).mean():.3f}")
