"""
Conditional DDPM with:
  - FiLM conditioning (scale/shift modulation, replaces concat)
  - Cosine noise schedule
  - v-prediction
  - Min-SNR-γ loss weighting (balances learning across noise levels)
  - Classifier-Free Guidance (multiple cfg_mode strategies)
  - Internal Y standardization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Building blocks
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
    """Feature-wise Linear Modulation: h_out = γ ⊙ h + β.

    条件信号通过生成 per-neuron 的 scale (γ) 和 shift (β) 来调制隐藏层.
    比 concat 强: concat 让条件和输入竞争同一组权重,
    FiLM 让条件直接控制每个神经元的激活幅度和偏移.
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        # 初始化为 identity modulation (γ=1, β=0), 训练初期不扰乱主干
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:hidden_dim] = 1.0  # γ init = 1

    def forward(self, h, cond):
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * h + beta


class FiLMResBlock(nn.Module):
    """Residual block with per-block FiLM conditioning + LayerNorm.

    h → LN → Linear → SiLU → FiLM(cond_i) → Linear → (+h)

    每个 block 有自己的 cond_proj，从 raw condition 中提取不同层次的信息。
    LayerNorm 稳定训练，防止梯度在深层网络中爆炸/消失。
    """
    def __init__(self, hidden_dim, cond_raw_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # 每个 block 自己的 cond projection
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


class NoisePredictor(nn.Module):
    """v/ε predictor with per-block FiLM conditioning.

    (y_t, x) → concat → proj_in → [FiLMResBlock × n_blocks] → proj_out → v̂

    改进:
    1. x concat 到 y_t 输入（不只依赖 FiLM 传递条件信息）
    2. 每个 block 有自己的 cond_proj（不同层提取不同条件特征）
    3. LayerNorm 稳定深层训练
    """
    def __init__(self, y_dim, x_dim, hidden_dim=256, t_embed_dim=64,
                 n_blocks=8):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(t_embed_dim)
        cond_raw_dim = t_embed_dim + x_dim

        # x 也 concat 到输入，双路条件化
        self.proj_in = nn.Sequential(
            nn.Linear(y_dim + x_dim, hidden_dim), nn.SiLU()
        )
        # 每个 block 从 raw cond 独立投影
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
# Noise schedule
# ═══════════════════════════════════════════════════════════════════

def cosine_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    alpha_bar = alpha_bar[:T]
    alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=1.0 - 1e-5)
    return alpha_bar.float()


# ═══════════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════════

class ConditionalDDPM(nn.Module):
    """Conditional DDPM: FiLM + Min-SNR-γ + v-prediction + CFG.

    改进点 vs 原版:
      1. FiLM conditioning: 条件通过 scale/shift 调制每层, 比 concat 更强
      2. Min-SNR-γ weighting: 平衡不同噪声水平的学习, 改善低噪声细节
      3. cfg_mode 参数: 多种 CFG 策略, 默认 "none" 适合回归分布
    """

    def __init__(self, y_dim, x_dim, T=200, hidden_dim=256, n_blocks=8,
                 schedule="cosine", beta_min=1e-4, beta_max=0.02,
                 cfg_drop_prob=0.15, min_snr_gamma=None):
        """
        Args:
            min_snr_gamma: Min-SNR-γ 裁剪值.
                w(t) = min(SNR(t), γ) / SNR(t)
                ⚠️ 仅适用于 ε-prediction. v-prediction 已隐式平衡 SNR,
                叠加 Min-SNR 会过度压低高噪声 step → 全局结构丢失.
                默认 None (关闭). 如需启用, 建议 γ=5.0 + ε-prediction.
            cfg_drop_prob: 训练时随机丢弃条件的概率 (用于 CFG).
        """
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.T = T
        self.cfg_drop_prob = cfg_drop_prob
        self.min_snr_gamma = min_snr_gamma

        # ── Noise schedule ──
        if schedule == "cosine":
            alpha_bar = cosine_schedule(T)
        else:
            betas = torch.linspace(beta_min, beta_max, T)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        # ── Min-SNR weights ──
        snr = alpha_bar / (1.0 - alpha_bar)  # [T]
        self.register_buffer("snr", snr)
        if min_snr_gamma is not None:
            min_snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr
        else:
            min_snr_weight = torch.ones_like(snr)
        self.register_buffer("min_snr_weight", min_snr_weight)

        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
        betas = 1.0 - alpha_bar / alpha_bar_prev
        self.register_buffer("betas", betas.clamp(max=0.999))
        self.register_buffer("alphas", 1.0 - self.betas)

        # ── Y normalization ──
        self.register_buffer("y_mean", torch.zeros(y_dim))
        self.register_buffer("y_std", torch.ones(y_dim))

        # ── Network ──
        self.noise_net = NoisePredictor(y_dim, x_dim, hidden_dim, n_blocks=n_blocks)

    # ── Normalization ──

    def set_normalization(self, y_mean, y_std):
        self.y_mean.copy_(y_mean)
        self.y_std.copy_(y_std.clamp(min=1e-6))

    def _normalize(self, y):
        return (y - self.y_mean) / self.y_std

    def _denormalize(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    # ── Diffusion math ──

    def q_sample(self, y_norm, t, eps=None):
        if eps is None:
            eps = torch.randn_like(y_norm)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * y_norm + sqrt_omab * eps, eps

    def _get_v_target(self, y_norm, eps, t):
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * eps - sqrt_omab * y_norm

    def _v_to_eps(self, v, y_t, t):
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * v + sqrt_omab * y_t

    # ── Training ──

    def training_loss(self, y_0, x):
        """v-prediction loss with Min-SNR-γ weighting + CFG dropout."""
        y_norm = self._normalize(y_0)
        B = y_norm.shape[0]
        t = torch.randint(0, self.T, (B,), device=y_norm.device)
        y_t, eps = self.q_sample(y_norm, t)
        v_target = self._get_v_target(y_norm, eps, t)

        # CFG: randomly drop conditioning
        x_input = x.clone()
        if self.training and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=x.device) < self.cfg_drop_prob
            x_input[drop_mask] = 0.0

        v_hat = self.noise_net(y_t, t.float() / self.T, x_input)

        # Per-sample MSE
        per_sample_loss = ((v_hat - v_target) ** 2).mean(dim=-1)  # [B]

        # Min-SNR-γ weighting: 提升低噪声 step 的权重
        weights = self.min_snr_weight[t]  # [B]
        loss = (weights * per_sample_loss).mean()

        return loss

    # ── Score (for conformal prediction) ──

    @torch.no_grad()
    def denoise_score(self, y, x, timesteps=None, n_repeats=5,
                      eps_bank=None, tau=None):
        """Denoising score with CRN, R-vectorized.

        For each timestep, all R repeats are stacked into the batch dimension
        and processed in a single forward pass: [B, yd] → [B*R, yd] → forward
        → reshape [B, R, yd] → sum over R.  This reduces the number of
        forward passes from T*R to T.

        Args:
            y: [B, y_dim] original scale
            x: [B, x_dim]
            timesteps: int list or LongTensor of timestep values
            n_repeats: noise samples per (y, t) — ignored if eps_bank given
            eps_bank: [n_timesteps, n_repeats, y_dim] pre-generated CRN noise.
                      If None, falls back to random sampling (legacy).
            tau: if provided, enable early rejection (usually None for speed).
        Returns:
            scores: [B] tensor
        """
        y_norm = self._normalize(y)
        B, yd, device = y_norm.shape[0], y_norm.shape[1], y_norm.device

        if timesteps is None:
            timesteps = torch.linspace(1, self.T - 1, 10).long().to(device)
        elif isinstance(timesteps, torch.Tensor):
            timesteps = timesteps.to(device).long()
        else:
            timesteps = torch.tensor(timesteps, device=device).long()

        n_ts = len(timesteps)
        if eps_bank is not None:
            assert eps_bank.shape[0] == n_ts, \
                f"eps_bank dim0 ({eps_bank.shape[0]}) != len(timesteps) ({n_ts})"
            R = eps_bank.shape[1]
        else:
            R = n_repeats
        n_total_terms = n_ts * R

        total_err = torch.zeros(B, device=device)

        # Pre-expand y and x for R repeats: [B, *] → [B*R, *]
        y_rep = y_norm.unsqueeze(1).expand(B, R, yd).reshape(B * R, yd)
        x_rep = x.unsqueeze(1).expand(B, R, x.shape[1]).reshape(B * R, x.shape[1])

        for t_idx, t_val in enumerate(timesteps):
            # Build eps: [B*R, yd]
            if eps_bank is not None:
                # eps_bank[t_idx]: [R, yd] → expand to [B, R, yd] → [B*R, yd]
                eps = eps_bank[t_idx].unsqueeze(0).expand(B, R, yd).reshape(B * R, yd)
            else:
                eps = torch.randn(B * R, yd, device=device)

            t_rep = t_val.expand(B * R)

            # Single forward pass for all B*R samples
            y_t, _ = self.q_sample(y_rep, t_rep, eps=eps)
            v_hat = self.noise_net(y_t, t_rep.float() / self.T, x_rep)
            eps_hat = self._v_to_eps(v_hat, y_t, t_rep)

            # Per-sample error: [B*R] → [B, R] → sum over R
            err = ((eps - eps_hat) ** 2).sum(-1)  # [B*R]
            total_err += err.reshape(B, R).sum(dim=1)  # [B]

        return total_err / n_total_terms

    # ── Sampling ──

    def _get_cfg_weight(self, step_idx, n_steps, cfg_scale, cfg_mode,
                        t_cur_normalized):
        if cfg_mode == "none" or cfg_scale == 1.0:
            return 1.0
        if cfg_mode == "static":
            return cfg_scale
        if cfg_mode == "dynamic":
            progress = step_idx / n_steps
            return 1.0 + (cfg_scale - 1.0) * (1.0 - progress)
        if cfg_mode == "low_temp":
            return cfg_scale if t_cur_normalized > 0.5 else 1.0
        return cfg_scale

    @torch.no_grad()
    def sample_ddim(self, x, n_steps=50, cfg_scale=1.0, cfg_mode="none"):
        """DDIM sampling.

        Args:
            cfg_scale: guidance strength. 1.0 = no guidance.
            cfg_mode: "none" | "dynamic" | "low_temp" | "static"
                回归分布推荐 "none"; 图像生成用 "static" + cfg_scale=2~7.5.
        """
        B, device = x.shape[0], x.device
        step_indices = torch.linspace(self.T - 1, 0, n_steps + 1).long()
        y_t = torch.randn(B, self.y_dim, device=device)
        x_uncond = torch.zeros_like(x)

        for i in range(n_steps):
            t_cur = step_indices[i].expand(B).to(device)
            t_next = step_indices[i + 1].expand(B).to(device)
            t_norm = t_cur.float() / self.T
            t_cur_normalized = float(step_indices[i]) / self.T

            w = self._get_cfg_weight(i, n_steps, cfg_scale, cfg_mode,
                                     t_cur_normalized)

            v_cond = self.noise_net(y_t, t_norm, x)
            if w != 1.0:
                v_uncond = self.noise_net(y_t, t_norm, x_uncond)
                v_hat = v_uncond + w * (v_cond - v_uncond)
            else:
                v_hat = v_cond

            eps_hat = self._v_to_eps(v_hat, y_t, t_cur)
            ab_cur = self.alpha_bar[t_cur].unsqueeze(-1)
            ab_next = self.alpha_bar[t_next.clamp(min=0)].unsqueeze(-1)
            y_pred = (y_t - torch.sqrt(1 - ab_cur) * eps_hat) / torch.sqrt(ab_cur)
            y_t = torch.sqrt(ab_next) * y_pred + torch.sqrt(1 - ab_next) * eps_hat

        return self._denormalize(y_t)

    @torch.no_grad()
    def sample_n(self, x_point, n, n_steps=100, batch_size=512):
        """Sample n points from p(y|x) for a single x.

        Args:
            x_point: [1, x_dim] or [x_dim]
            n: number of samples
            n_steps: DDIM steps
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
            yb = self.sample_ddim(xb, n_steps=n_steps)
            samples.append(yb)
        return torch.cat(samples, dim=0)  # [n, y_dim]

    # ── Encoding (data → noise via DDIM forward) ──

    @torch.no_grad()
    def encode(self, x, y, n_steps=50, solver="midpoint"):
        """Map data y to latent z via DDIM deterministic encoding.

        DDIM encoding goes from t=0 (clean data) to t=T (noise).
        If the model is well-trained, z ≈ N(0, I).

        Args:
            x: [B, x_dim]
            y: [B, y_dim] original scale
            n_steps: encoding steps
            solver: unused (kept for API compatibility with FM)

        Returns:
            z: [B, y_dim] latent codes
        """
        y_norm = self._normalize(y)
        B, device = y_norm.shape[0], y_norm.device

        step_indices = torch.linspace(0, self.T - 1, n_steps + 1).long()
        y_t = y_norm.clone()

        for i in range(n_steps):
            t_cur = step_indices[i].expand(B).to(device)
            t_next = step_indices[i + 1].expand(B).to(device)
            t_norm = t_cur.float() / self.T

            v_hat = self.noise_net(y_t, t_norm, x)
            eps_hat = self._v_to_eps(v_hat, y_t, t_cur)

            ab_cur = self.alpha_bar[t_cur.clamp(max=self.T-1)].unsqueeze(-1)
            ab_next = self.alpha_bar[t_next.clamp(max=self.T-1)].unsqueeze(-1)

            # DDIM forward: predict y_0, then re-noise to t_next
            y_pred = (y_t - torch.sqrt(1 - ab_cur) * eps_hat) / torch.sqrt(ab_cur)
            y_t = torch.sqrt(ab_next) * y_pred + torch.sqrt(1 - ab_next) * eps_hat

        return y_t  # at t=T, should be ~N(0,I)
