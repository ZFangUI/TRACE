"""
Conditional Flow Matching (OT-CFM) with:
  - FiLM conditioning (scale/shift modulation, replaces concat)
  - Loss-aware timestep sampling (focuses training on hard timesteps)
  - Midpoint ODE solver (2nd-order, more accurate than Euler at same cost)
  - Classifier-Free Guidance (multiple cfg_mode strategies)
  - Internal Y standardization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Building blocks (same FiLM architecture as diffusion.py)
# ═══════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=device) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h_out = γ ⊙ h + β."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:hidden_dim] = 1.0

    def forward(self, h, cond):
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * h + beta


class FiLMResBlock(nn.Module):
    """Residual block with per-block FiLM conditioning + LayerNorm."""
    def __init__(self, hidden_dim, cond_raw_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film = FiLMLayer(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, h, cond_raw):
        residual = h
        cond = self.cond_proj(cond_raw)
        h = self.norm(h)
        h = self.act(self.linear1(h))
        h = self.film(h, cond)
        h = self.linear2(h)
        return self.act(h + residual)


class VelocityNet(nn.Module):
    """Velocity predictor with per-block FiLM conditioning.

    (y_t, x) → concat → proj_in → [FiLMResBlock × n_blocks] → proj_out → v̂
    """
    def __init__(self, y_dim, x_dim, hidden_dim=256, t_embed_dim=64,
                 n_blocks=8):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(t_embed_dim)
        cond_raw_dim = t_embed_dim + x_dim

        self.proj_in = nn.Sequential(
            nn.Linear(y_dim + x_dim, hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim, cond_raw_dim) for _ in range(n_blocks)
        ])
        self.proj_out = nn.Linear(hidden_dim, y_dim)

    def forward(self, y_t, t, x):
        t_emb = self.time_embed(t)
        cond_raw = torch.cat([t_emb, x], dim=-1)
        h = self.proj_in(torch.cat([y_t, x], dim=-1))
        for block in self.blocks:
            h = block(h, cond_raw)
        return self.proj_out(h)


# ═══════════════════════════════════════════════════════════════════
# Loss-aware timestep sampler
# ═══════════════════════════════════════════════════════════════════

class LossAwareTimestepSampler:
    """维护每个 t-bin 的 EMA loss, 按 loss 比例采样 t.

    原理: loss 高的 t (模型预测差的区间) 被更频繁采样,
    加速这些区间的学习. 相比均匀采样, 这让 spiral 尾部等
    难学的区域获得更多训练信号.

    实现: 将 [0, 1] 分成 n_bins 个桶, 每个桶维护 EMA loss,
    采样概率 ∝ sqrt(loss_bucket) (用 sqrt 避免过度集中).
    """
    def __init__(self, n_bins=100, ema_decay=0.999, warmup=500):
        """
        Args:
            n_bins: t 区间桶数
            ema_decay: loss EMA 衰减系数
            warmup: 前 warmup 步用均匀采样, 等 loss 统计稳定
        """
        self.n_bins = n_bins
        self.ema_decay = ema_decay
        self.warmup = warmup
        self.loss_ema = np.ones(n_bins)  # 初始化为均匀
        self.step_count = 0

    def sample(self, batch_size, device):
        """采样 t ∈ [0, 1].

        Returns:
            t: [B] tensor
        """
        self.step_count += 1

        if self.step_count < self.warmup:
            # Warmup: 均匀采样
            return torch.rand(batch_size, device=device)

        # 按 sqrt(loss) 比例采样 bin
        probs = np.sqrt(self.loss_ema)
        probs = probs / probs.sum()
        bins = np.random.choice(self.n_bins, size=batch_size, p=probs)

        # 在 bin 内均匀采样
        t_lo = bins / self.n_bins
        t_hi = (bins + 1) / self.n_bins
        t_np = np.random.uniform(t_lo, t_hi)
        return torch.tensor(t_np, dtype=torch.float32, device=device)

    def update(self, t, per_sample_loss):
        """用当前 batch 的 loss 更新 EMA.

        Args:
            t: [B] tensor, 采样的 t 值
            per_sample_loss: [B] tensor, 每个样本的 loss
        """
        t_np = t.detach().cpu().numpy()
        loss_np = per_sample_loss.detach().cpu().numpy()
        bins = np.clip((t_np * self.n_bins).astype(int), 0, self.n_bins - 1)

        for b in range(self.n_bins):
            mask = bins == b
            if mask.sum() > 0:
                mean_loss = loss_np[mask].mean()
                self.loss_ema[b] = (self.ema_decay * self.loss_ema[b]
                                    + (1 - self.ema_decay) * mean_loss)


# ═══════════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════════

class ConditionalFlowMatching(nn.Module):
    """Conditional OT-CFM: FiLM + loss-aware sampling + midpoint ODE.

    改进点 vs 原版:
      1. FiLM conditioning: 条件通过 scale/shift 调制, 比 concat 更强
      2. Loss-aware t sampling: 难学的 t 区间获得更多训练 (通过外部 sampler)
      3. Midpoint ODE solver: 2 阶精度, 相同步数下比 Euler 更准确
      4. cfg_mode: 多种 CFG 策略, 默认 "none"
    """

    def __init__(self, y_dim, x_dim, hidden_dim=256, n_blocks=8, sigma_min=1e-4,
                 cfg_drop_prob=0.15):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.sigma_min = sigma_min
        self.cfg_drop_prob = cfg_drop_prob

        self.register_buffer("y_mean", torch.zeros(y_dim))
        self.register_buffer("y_std", torch.ones(y_dim))

        self.velocity_net = VelocityNet(y_dim, x_dim, hidden_dim, n_blocks=n_blocks)

        # Loss-aware timestep sampler (在 training.py 中使用)
        self.t_sampler = LossAwareTimestepSampler()

    # ── Normalization ──

    def set_normalization(self, y_mean, y_std):
        self.y_mean.copy_(y_mean)
        self.y_std.copy_(y_std.clamp(min=1e-6))

    def _normalize(self, y):
        return (y - self.y_mean) / self.y_std

    def _denormalize(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    # ── Training ──

    def training_loss(self, y_1, x):
        """OT-CFM loss with loss-aware t sampling + CFG dropout."""
        y_1_norm = self._normalize(y_1)
        B, device = y_1_norm.shape[0], y_1_norm.device

        # Loss-aware timestep sampling
        t = self.t_sampler.sample(B, device)

        y_0 = torch.randn_like(y_1_norm)
        t_exp = t.unsqueeze(-1)
        y_t = (1 - t_exp) * y_0 + t_exp * y_1_norm
        u_t = y_1_norm - y_0

        # CFG: randomly drop conditioning
        x_input = x.clone()
        if self.training and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob
            x_input[drop_mask] = 0.0

        v_hat = self.velocity_net(y_t, t, x_input)

        # Per-sample loss (for updating t_sampler)
        per_sample_loss = ((v_hat - u_t) ** 2).mean(dim=-1)  # [B]

        # Update loss-aware sampler
        self.t_sampler.update(t, per_sample_loss)

        return per_sample_loss.mean()

    # ── Score (for conformal prediction) ──

    @torch.no_grad()
    def path_score(self, y, x, n_timesteps=10, n_repeats=5,
                   timesteps=None, y0_bank=None, tau=None):
        """Path consistency score with CRN, R-vectorized.

        For each timestep, all R repeats are stacked into the batch dimension
        and processed in a single forward pass: [B, yd] → [B*R, yd] → forward
        → reshape [B, R, yd] → sum over R.  This reduces the number of
        forward passes from T*R to T.

        Args:
            y: [B, y_dim] original scale
            x: [B, x_dim]
            n_timesteps: number of t values (ignored if timesteps given)
            n_repeats: source samples per t (ignored if y0_bank given)
            timesteps: float list or FloatTensor. If None, linspace(0.01, 0.99).
            y0_bank: [n_timesteps, n_repeats, y_dim] pre-generated CRN noise.
                     If None, falls back to random sampling (legacy).
            tau: if provided, enable early rejection (usually None for speed).
        Returns:
            scores: [B] tensor
        """
        y_norm = self._normalize(y)
        B, yd, device = y_norm.shape[0], y_norm.shape[1], y_norm.device

        if timesteps is None:
            ts = torch.linspace(0.01, 0.99, n_timesteps, device=device)
        elif isinstance(timesteps, torch.Tensor):
            ts = timesteps.to(device=device, dtype=torch.float32)
        else:
            ts = torch.tensor(timesteps, device=device, dtype=torch.float32)

        n_ts = len(ts)
        if y0_bank is not None:
            assert y0_bank.shape[0] == n_ts, \
                f"y0_bank dim0 ({y0_bank.shape[0]}) != len(timesteps) ({n_ts})"
            R = y0_bank.shape[1]
        else:
            R = n_repeats
        n_total_terms = n_ts * R

        total_err = torch.zeros(B, device=device)

        # Pre-expand y and x for R repeats: [B, *] → [B*R, *]
        y_rep = y_norm.unsqueeze(1).expand(B, R, yd).reshape(B * R, yd)
        x_rep = x.unsqueeze(1).expand(B, R, x.shape[1]).reshape(B * R, x.shape[1])

        for t_idx, t_val in enumerate(ts):
            # Build y0: [B*R, yd]
            if y0_bank is not None:
                # y0_bank[t_idx]: [R, yd] → expand to [B, R, yd] → [B*R, yd]
                y_0 = y0_bank[t_idx].unsqueeze(0).expand(B, R, yd).reshape(B * R, yd)
            else:
                y_0 = torch.randn(B * R, yd, device=device)

            t_rep = t_val.expand(B * R)
            t_exp = t_rep.unsqueeze(-1)  # [B*R, 1]

            # Single forward pass for all B*R samples
            y_t = (1 - t_exp) * y_0 + t_exp * y_rep
            u_t = y_rep - y_0
            v_hat = self.velocity_net(y_t, t_rep, x_rep)

            # Per-sample error: [B*R] → [B, R] → sum over R
            err = ((v_hat - u_t) ** 2).sum(-1)  # [B*R]
            total_err += err.reshape(B, R).sum(dim=1)  # [B]

        return total_err / n_total_terms

    # ── Sampling ──

    def _get_cfg_weight(self, t_val, cfg_scale, cfg_mode):
        if cfg_mode == "none" or cfg_scale == 1.0:
            return 1.0
        if cfg_mode == "static":
            return cfg_scale
        if cfg_mode == "dynamic":
            # FM: t 从 0 (noise) 到 1 (data), 前期高 guidance, 后期衰减
            return 1.0 + (cfg_scale - 1.0) * (1.0 - t_val)
        if cfg_mode == "low_temp":
            return cfg_scale if t_val < 0.5 else 1.0
        return cfg_scale

    def _velocity_with_cfg(self, y_t, t, x, x_uncond, w):
        """Compute velocity with optional CFG."""
        v_cond = self.velocity_net(y_t, t, x)
        if w != 1.0:
            v_uncond = self.velocity_net(y_t, t, x_uncond)
            return v_uncond + w * (v_cond - v_uncond)
        return v_cond

    @torch.no_grad()
    def sample(self, x, n_steps=50, cfg_scale=1.0, cfg_mode="none",
               solver="midpoint"):
        """ODE sampling with configurable solver and CFG.

        Args:
            x: conditioning input [B, x_dim]
            n_steps: number of ODE steps
            cfg_scale: guidance strength. 1.0 = no guidance.
            cfg_mode: "none" | "dynamic" | "low_temp" | "static"
            solver: ODE solver
                "euler"    — 1st order, 最快, 每步 1 次 velocity 评估
                "midpoint" — 2nd order, 每步 2 次评估, 精度显著提升 (推荐)

        推荐:
            回归分布: solver="midpoint", n_steps=50~100, cfg_mode="none"
            快速测试: solver="euler", n_steps=100~200
        """
        B, device = x.shape[0], x.device
        dt = 1.0 / n_steps
        y_t = torch.randn(B, self.y_dim, device=device)
        x_uncond = torch.zeros_like(x)

        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device)
            w = self._get_cfg_weight(t_val, cfg_scale, cfg_mode)

            if solver == "midpoint":
                # ── Midpoint method (2nd order) ──
                # k1 = v(y_t, t)
                k1 = self._velocity_with_cfg(y_t, t, x, x_uncond, w)

                # k2 = v(y_t + 0.5*dt*k1, t + 0.5*dt)
                y_mid = y_t + 0.5 * dt * k1
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=device)
                w_mid = self._get_cfg_weight(t_val + 0.5 * dt, cfg_scale,
                                             cfg_mode)
                k2 = self._velocity_with_cfg(y_mid, t_mid, x, x_uncond, w_mid)

                y_t = y_t + dt * k2

            else:  # euler
                v = self._velocity_with_cfg(y_t, t, x, x_uncond, w)
                y_t = y_t + v * dt

        return self._denormalize(y_t)

    @torch.no_grad()
    def sample_n(self, x_point, n, n_steps=100, batch_size=512):
        """Sample n points from p(y|x) for a single x.

        Args:
            x_point: [1, x_dim] or [x_dim]
            n: number of samples
            n_steps: ODE steps
            batch_size: max batch to avoid OOM

        Returns:
            [n, y_dim]
        """
        if x_point.dim() == 1:
            x_point = x_point.unsqueeze(0)
        device = next(self.parameters()).device
        x_point = x_point.to(device)
        samples = []
        for i in range(0, n, batch_size):
            b = min(batch_size, n - i)
            xb = x_point.expand(b, -1)
            yb = self.sample(xb, n_steps=n_steps)
            samples.append(yb)
        return torch.cat(samples, dim=0)  # [n, y_dim]

    # ── Encoding (data → noise via reverse ODE) ──

    @torch.no_grad()
    def encode(self, x, y, n_steps=50, solver="midpoint"):
        """Map data y to latent z via reverse ODE integration.

        The forward ODE goes from noise (t=0) to data (t=1):
            dy/dt = v_θ(y_t, t, x)

        To encode, we reverse: from y at t=1, integrate back to t=0.
        If the model is well-trained, z ≈ N(0, I).

        Args:
            x: [B, x_dim]
            y: [B, y_dim] original scale
            n_steps: ODE integration steps
            solver: "euler" or "midpoint"

        Returns:
            z: [B, y_dim] latent codes
        """
        y_norm = self._normalize(y)
        B, device = y_norm.shape[0], y_norm.device
        dt = 1.0 / n_steps

        z = y_norm.clone()

        for i in range(n_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=device)

            if solver == "midpoint":
                k1 = self.velocity_net(z, t, x)
                z_mid = z - 0.5 * dt * k1
                t_mid = torch.full((B,), t_val - 0.5 * dt, device=device)
                k2 = self.velocity_net(z_mid, t_mid, x)
                z = z - dt * k2
            else:
                v = self.velocity_net(z, t, x)
                z = z - dt * v

        return z
