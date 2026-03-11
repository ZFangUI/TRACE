"""
Data splitting: Train / Calibration / Test.
"""

import torch


def split_data(X, Y, seed=0, train_frac=0.675, cal_frac=0.225):
    """Split into train/cal/test.

    Default: 67.5% train, 22.5% cal, 10% test.
    """
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    n_train = int(n * train_frac)
    n_cal = int(n * cal_frac)

    tr = perm[:n_train]
    cal = perm[n_train:n_train + n_cal]
    te = perm[n_train + n_cal:]

    return {
        "trx": X[tr], "try": Y[tr],
        "cx": X[cal], "cy": Y[cal],
        "tsx": X[te], "tsy": Y[te],
    }
