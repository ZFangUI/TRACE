"""
Training functions for NF, Diffusion, and Flow Matching models.

Features:
  - Early stopping with patience (val loss on held-out 10% of training data)
  - Warmup + cosine annealing LR schedule
  - EMA for Diff/FM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import math


def clear_gpu():
    """Force garbage collection and release cached GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _split_val(train_x, train_y, val_frac=0.1, seed=0):
    """Split off a validation set from training data."""
    n = train_x.shape[0]
    n_val = max(1, int(n * val_frac))
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return (train_x[train_idx], train_y[train_idx],
            train_x[val_idx], train_y[val_idx])


def _make_warmup_cosine_scheduler(optimizer, epochs, warmup_frac=0.05):
    """Linear warmup then cosine annealing to 0."""
    warmup_epochs = max(1, int(epochs * warmup_frac))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        # cosine decay from 1 to 0 over remaining epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════════
# NF Training
# ═════════════════════════════════════════════════════════════════

def train_nf(model, train_x, train_y, epochs=200, batch_size=256,
             lr=1e-3, weight_decay=1e-5, grad_clip=1.0,
             patience=0, device="cpu", verbose=True):
    """Train conditional NF via NLL.

    Args:
        patience: early stopping patience (0 = disabled).
    """
    model = model.to(device)

    # Validation split for early stopping
    if patience > 0:
        trx, try_, vx, vy = _split_val(train_x, train_y)
        vx, vy = vx.to(device), vy.to(device)
    else:
        trx, try_ = train_x, train_y

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _make_warmup_cosine_scheduler(opt, epochs)

    loader = DataLoader(
        TensorDataset(trx.to(device), try_.to(device)),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for xb, yb in loader:
            z, log_det = model(xb, yb)
            nll = 0.5 * (z ** 2).sum(-1) - log_det
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()

        if verbose and ep % max(1, epochs // 10) == 0:
            print(f"  [NF]   Epoch {ep:4d}/{epochs}  NLL={ep_loss/nb:.4f}")

        # Early stopping check (skip first 20% epochs as grace period)
        if patience > 0 and ep >= max(1, int(epochs * 0.2)):
            model.eval()
            with torch.no_grad():
                z, ld = model(vx, vy)
                val_loss = (0.5 * (z ** 2).sum(-1) - ld).mean().item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  [NF]   Early stop at epoch {ep} "
                              f"(best val={best_val_loss:.4f})")
                    break

    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)

    model.cpu()
    del opt, sched, loader
    clear_gpu()
    return model


# ═════════════════════════════════════════════════════════════════
# Diffusion Training
# ═════════════════════════════════════════════════════════════════

def train_diffusion(model, train_x, train_y, epochs=300, batch_size=256,
                    lr=1e-3, weight_decay=0.0, grad_clip=1.0,
                    ema_decay=0.999, patience=0,
                    device="cpu", verbose=True):
    """Train conditional DDPM with EMA and optional early stopping."""
    model = model.to(device)

    # Validation split
    if patience > 0:
        trx, try_, vx, vy = _split_val(train_x, train_y)
        vx, vy = vx.to(device), vy.to(device)
    else:
        trx, try_ = train_x, train_y

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _make_warmup_cosine_scheduler(opt, epochs)

    # EMA
    ema_model = copy.deepcopy(model)

    loader = DataLoader(
        TensorDataset(trx.to(device), try_.to(device)),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    best_val_loss = float('inf')
    best_ema_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for xb, yb in loader:
            loss = model.training_loss(yb, xb)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            # EMA update
            with torch.no_grad():
                for p, ep_p in zip(model.parameters(), ema_model.parameters()):
                    ep_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            ep_loss += loss.item()
            nb += 1
        sched.step()

        if verbose and ep % max(1, epochs // 10) == 0:
            print(f"  [Diff] Epoch {ep:4d}/{epochs}  loss={ep_loss/nb:.6f}")

        # Early stopping on EMA model's validation loss
        # Skip first 20% of epochs (warmup grace period)
        if patience > 0 and ep >= max(1, int(epochs * 0.2)):
            ema_model.eval()
            with torch.no_grad():
                val_loss = sum(ema_model.training_loss(vy, vx).item() for _ in range(5)) / 5
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ema_state = copy.deepcopy(ema_model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  [Diff] Early stop at epoch {ep} "
                              f"(best val={best_val_loss:.6f})")
                    break

    # Load best EMA weights (or final EMA if no early stopping)
    if patience > 0 and best_ema_state is not None:
        model.load_state_dict(best_ema_state)
    else:
        model.load_state_dict(ema_model.state_dict())

    model.cpu()
    del opt, sched, ema_model, loader
    clear_gpu()
    return model


# ═════════════════════════════════════════════════════════════════
# Flow Matching Training
# ═════════════════════════════════════════════════════════════════

def train_flow_matching(model, train_x, train_y, epochs=300, batch_size=256,
                        lr=1e-3, weight_decay=0.0, grad_clip=1.0,
                        ema_decay=0.999, patience=0,
                        device="cpu", verbose=True):
    """Train OT-CFM with EMA and optional early stopping."""
    model = model.to(device)

    # Validation split
    if patience > 0:
        trx, try_, vx, vy = _split_val(train_x, train_y)
        vx, vy = vx.to(device), vy.to(device)
    else:
        trx, try_ = train_x, train_y

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _make_warmup_cosine_scheduler(opt, epochs)

    # EMA
    ema_model = copy.deepcopy(model)

    loader = DataLoader(
        TensorDataset(trx.to(device), try_.to(device)),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )

    best_val_loss = float('inf')
    best_ema_state = None
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for xb, yb in loader:
            loss = model.training_loss(yb, xb)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            # EMA update
            with torch.no_grad():
                for p, ep_p in zip(model.parameters(), ema_model.parameters()):
                    ep_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            ep_loss += loss.item()
            nb += 1
        sched.step()

        if verbose and ep % max(1, epochs // 10) == 0:
            print(f"  [FM]   Epoch {ep:4d}/{epochs}  loss={ep_loss/nb:.6f}")

        # Early stopping on EMA model's validation loss
        # Skip first 20% of epochs (warmup grace period)
        if patience > 0 and ep >= max(1, int(epochs * 0.2)):
            ema_model.eval()
            with torch.no_grad():
                val_loss = sum(ema_model.training_loss(vy, vx).item() for _ in range(5)) / 5
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ema_state = copy.deepcopy(ema_model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  [FM]   Early stop at epoch {ep} "
                              f"(best val={best_val_loss:.6f})")
                    break

    # Load best EMA weights (or final EMA if no early stopping)
    if patience > 0 and best_ema_state is not None:
        model.load_state_dict(best_ema_state)
    else:
        model.load_state_dict(ema_model.state_dict())

    model.cpu()
    del opt, sched, ema_model, loader
    clear_gpu()
    return model
