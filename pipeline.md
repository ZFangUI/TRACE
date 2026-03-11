# Pipeline 说明

## 概览

本项目实现了基于生成模型（Normalizing Flow、Diffusion、Flow Matching）的 Conformal Prediction Region 构造框架。完整流程分为 **训练 → 评估 → 绘图** 三个阶段，其中训练和绘图完全解耦。

---

## 1. 训练 + 评估（experiment.py）

### 基本用法

```bash
# 单次运行（调参用，结果存到 _single 目录）
python experiment.py --dataset pinwheel --n_total 30000

# 多次重复（正式实验，报告 mean±std）
python experiment.py --dataset pinwheel --n_total 30000 --n_repeats 10
```

### 存储路径

```
n_repeats == 1  →  experiments/{dataset}_s{seed}_single/
n_repeats >  1  →  experiments/{dataset}_s{seed}/
```

两者互不覆盖，调参不影响正式结果。

### 内部流程

```
experiment.py
│
├── 1. 数据生成 + 划分
│   └── train / calibration / test (6:2:2)
│
├── 2. 模型训练（按需，只训练所选方法依赖的模型）
│   ├── NF  (RealNVP / NSF)     → nf_model.pt
│   ├── Diff (DDPM + FiLM)      → diff_model.pt
│   └── FM  (OT-CFM + FiLM)     → fm_model.pt
│
├── 3. Score 计算 + Conformal 校准
│   ├── Z-space: NF-Ball, NF-NLL, Diff-Denoise, FM-Path, ...
│   └── Baselines: RCP, NLE, PCP-Diff, MCQR, DistSplit, CQR
│
├── 4. 评估（Coverage, Volume）
│
├── 5. 保存
│   ├── repeat_000/
│   │   ├── nf_model.pt, diff_model.pt, fm_model.pt
│   │   ├── data_split.pt
│   │   ├── baselines.pt        ← baseline 对象（fit 后的状态）
│   │   └── results.json        ← coverage, tau, volume
│   └── config.json             ← 实验参数（供重绘脚本读取）
│
└── 6. 草稿图（快速预览）
    ├── region plot
    ├── comparison table
    └── sample quality plot
```

### 保存的文件说明

| 文件 | 内容 | 用途 |
|------|------|------|
| `config.json` | 所有实验参数 | single_plot.py 读取 dataset 名、score 参数等 |
| `nf/diff/fm_model.pt` | 模型权重 | single_plot.py 加载模型计算 grid score |
| `data_split.pt` | 训练/校准/测试数据 + Y 归一化参数 | single_plot.py 获取测试点和 scatter 数据 |
| `baselines.pt` | fit 好的 baseline 对象 (RCP, NLE, MCQR, PCP 等) | single_plot.py 直接加载，无需重新 fit |
| `results.json` | 每个方法的 coverage, tau, volume | single_plot.py 读取 tau，replot.py 画 violin |

---

## 2. 正式绘图

### 2.1 Region Plot（single_plot.py）

从 checkpoint 加载模型和 baseline，重新计算 grid score，画干净的 region plot。

```bash
python single_plot.py --exp_dir ./experiments/pinwheel_s0_single --x_index 750
python single_plot.py --exp_dir ./experiments/twomoons_s0_single --x_index 500 --device cuda
```

**输出：** `{exp_dir}/replot/`

```
replot/
├── pinwheel_regions_x750_combined.png    # 合并大图
├── pinwheel_ddpm_x750.png               # 各方法单独图
├── pinwheel_fm_x750.png
├── pinwheel_contra_x750.png
├── pinwheel_japan_x750.png
├── pinwheel_pcp_diff_x750.png
├── pinwheel_mcqr_x750.png
├── pinwheel_rcp_x750.png
├── pinwheel_nle_x750.png
└── pinwheel_true_density_x750.png
```

**特点：**
- 标题只有方法名（无 cov/τ/vol）
- 无坐标轴刻度
- 方法按固定顺序排列：DDPM → FM → CONTRA → JAPAN → PCP-Diff → MCQR → RCP → NLE
- 自动排除：DistSplit, Diff-ODE-Ball, FM-ODE-Ball, CQR
- 模型架构从 state_dict 自动推断（兼容旧 checkpoint）

### 2.2 Violin Plot（replot.py）

从 results.json 读取多次重复结果，画 coverage 和 volume 的 violin 图。

```bash
python replot.py --exp_dir ./experiments/pinwheel_s0
```

**输出：** `{exp_dir}/replot/`

```
replot/
├── pinwheel_coverage_violin.png
└── pinwheel_volume_violin.png
```

**注意：** 需要 `n_repeats ≥ 2` 才能画 violin（单次没有分布）。

---

## 3. 方法命名对照

| 内部名 | 显示名 | Score 函数 | 模型 |
|--------|--------|-----------|------|
| NF-Ball | CONTRA | s = ‖z‖² | NF |
| NF-NLL | JAPAN | s = 0.5‖z‖² − log\|det J\| | NF |
| Diff-Denoise | DDPM | s = E[‖ε − ε̂‖²] | Diffusion |
| FM-Path | FM | s = E[‖v − v̂‖²] | Flow Matching |
| RCP | RCP | Mahalanobis distance | NF (samples) |
| NLE | NLE | Local Mahalanobis | NF (samples) |
| PCP-Diff | PCP-Diff | kNN distance to samples | Diffusion (samples) |
| MCQR | MCQR | Weighted quantile box | NF (samples) |

---

## 4. 常用命令参考

### 快速调参（单次）

```bash
# NF only
python experiment.py --dataset pinwheel --n_total 30000 \
  --nf_flow_type nsf --nf_n_bins 9 --nf_n_layers 10 --nf_epochs 250 --nf_lr 1e-4 \
  --methods NF-Ball,NF-NLL --no_baselines

# 全部方法
python experiment.py --dataset pinwheel --n_total 30000 \
  --nf_flow_type nsf --nf_n_bins 9 --nf_n_layers 10 --nf_epochs 250 --nf_lr 1e-4 \
  --diff_epochs 2000 --fm_epochs 2000 --diff_patience 0 --fm_patience 0 \
  --diff_n_blocks 10 --fm_n_blocks 10 \
  --diff_score_timesteps 30 --diff_score_repeats 20 \
  --fm_score_timesteps 30 --fm_score_repeats 20
```

### 正式实验（多次重复）

```bash
python experiment.py --dataset twomoons --n_total 30000 --n_repeats 10 \
  --nf_flow_type nsf --nf_n_bins 9 --nf_n_layers 10 --nf_epochs 250 --nf_lr 1e-4 \
  --diff_epochs 2000 --fm_epochs 2000 --diff_patience 0 --fm_patience 0
```

### 绘图

```bash
# Region plot（单次实验）
python single_plot.py --exp_dir ./experiments/twomoons_s0_single --x_index 750 --device cuda

# Violin plot（多次实验）
python replot.py --exp_dir ./experiments/twomoons_s0
```

---

## 5. 调参建议

| 参数 | 建议范围 | 说明 |
|------|---------|------|
| `n_total` | 15000–30000 | 多模态数据集需要更多数据 |
| `nf_flow_type` | nsf | NSF 比 RealNVP 表达力强 |
| `nf_n_bins` | 7–12 | 太多过拟合，太少欠拟合 |
| `nf_n_layers` | 8–12 | |
| `nf_lr` | 1e-4 | NSF 需要较小学习率 |
| `nf_epochs` | 200–500 | NSF 收敛快 |
| `diff/fm_n_blocks` | 8–10 | 数据量 < 30k 不要超过 12 |
| `diff/fm_score_timesteps` | 15–30 | 更多 → 更低 MC 方差 → 更小 volume |
| `diff/fm_score_repeats` | 8–20 | 同上，收益递减 |
| `hidden_dim` | 256 | 保持默认，增大容易过拟合 |
