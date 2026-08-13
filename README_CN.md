# Spk-Res

[English](./README_EN.md) | **简体中文**

## 简介

**Spk-Res** 旨在利用深度学习方法修复高密度阵列电极（high-density probe）中损坏通道（bad channel）上的神经信号数据。

高密度阵列电极采集到的信号会在相邻电极之间留下空间印迹，因此通道间存在可被利用的空间关系。传统上，这类坏道插补通常采用克里金插值（Kriging）方法。本项目试图回答以下问题：

1. **Kriging 能否捕捉这种非线性空间关系？** 特别是当主通道信号缺失时，重建能否在记录过程中实时进行？
2. **深度学习模型的表现是否优于传统方法？** 我们对比了 Kriging 与基于深度学习的重建方法，验证后者对损失通道上神经元波形的重建精度。
3. **模型学到了什么？** 我们借助 **LAM（Local Attribution Map，局部归因图）** 方法分析模型：通过反向传播（梯度 / 路径梯度等）生成局部归因图，量化模型重建特定通道 / 时间窗口时对输入区域的依赖程度，并对比不同模糊路径（如线性插值、高斯模糊路径）与归因策略的影响，从而理解模型依赖的空间特征来源。

## 数据集

| 数据集 | 用途 | 来源 |
| --- | --- | --- |
| Overlapping Datasets | 急性记录常规混叠场景下重建能力评估 / 验证 | [DANDI 000034](https://dandiarchive.org/dandiset/000034) |
| Drifting Datasets | 漂移场景下重建能力评估 | [Steinmetz et al., Science 2021](https://figshare.com/articles/dataset/_Imposed_motion_datasets_from_Steinmetz_et_al_Science_2021/14024495) |
| Transfer Evaluation Dataset | 迁移评估（小规模数据集） | [Zenodo 3696926](https://zenodo.org/records/3696926) |

## 环境安装

使用 `conda` 创建环境并安装依赖：

```bash
conda create -n spk_res python=3.9
conda activate spk_res
pip install -r requirements.txt
```

> 本项目基于 [BasicSR](https://github.com/XPixelGroup/BasicSR) 重构，核心依赖包括 PyTorch、SpikeInterface、MEArec、KiloSort、MountainSort5 等神经科学与超分辨率工具链。

## 目录结构

```
Spk-Res/
├── main_binfile_v2.py      # .bin 格式记录：数据生成 → 训练 → 重建
├── main_meafile_v2.py      # .h5 (MEArec) 格式记录：完整流程 + spike sorter 评估
├── main_nwbfile_v2.py      # .nwb 格式记录入口
├── test_sorter_h5.py       # 调用 spike sorter 评估重建质量
├── res_utils/              # 核心工具库（数据生成、训练、评估、LAM 归因）
├── basicsr/                # 基于 BasicSR 的模型与训练框架
├── options_linux/          # 训练 / 测试配置文件（.yml）
├── options_linux_list/     # 批量实验配置（含持续学习 / 蒸馏等变体）
├── analysis/               # 结果分析脚本与 Notebook
├── tables/                 # 各方法、各 sorter 的评估结果表（F1 / Precision / Recall）
└── LAM/                    # 局部归因图（Local Attribution Map）分析
```

## 快速开始

### 1. 配置实验参数

所有参数均通过 `options_linux/` 与 `options_linux_list/` 目录下的 YAML 配置文件设定，主要包括：

- **数据生成设置**（`settings`）：下采样因子 `factor`、是否启用 Kriging 渲染（`is_krig`）、滑窗大小 `slide_window`、训练 / 测试数据比例等；
- **网络结构**（`network_g`）：支持 EDSR 与 Restormer 两种骨干网络；
- **训练设置**（`train`）：优化器、学习率调度、损失函数（MSE / L1）等；
- **验证与日志**（`val` / `logger`）。

### 2. 运行流程

以 `.h5` 格式记录为例（`main_meafile_v2.py`）：

```bash
python main_meafile_v2.py
```

完整流程分为四步：

1. **生成训练数据** —— 对原始记录做坏道掩码（损坏通道模拟）并渲染训练样本；
2. **训练模型** —— 在生成的训练集上训练重建网络；
3. **重建录音** —— 恢复损坏通道，并计算 MSE / NRMSE 等指标；
4. **下游评估** —— 用 spike sorter（如 MountainSort5、HerdingSpikes）对重建结果做神经元聚类，统计命中率（hit rate）与 F1 / Precision / Recall。

> 注意：脚本中的路径均为示例，请根据实际数据位置修改 `options_linux/test_bin_demo.yml` 中的 `settings.rec_path`、`save_path`、`weights_path` 等字段。

## 实验与发现

### 1. Overlapping Datasets

- 在急性记录中，仅使用前 30% 的数据即可重建后 70% 的数据；
- 相比 Kriging，深度学习方法对损坏通道上的神经元波形重建精度更高；
- 损坏通道越多，重建难度越大，误差越高。

### 2. Drifting Datasets

- 将其他电极上预训练好的权重**直接 zero-shot** 到相同漂移模式的数据上，可以将 MSE 压到很低；但 **MSE 低并不等于重建效果好**——需要结合下游 spike sorter 的聚类质量综合评估。

### 3. Transfer Evaluation Dataset

- 与前两个数据集相比，该数据集需要更高的训练数据占比。主要原因是该数据集时长很短（300 s）：若训练占比 30% 仅约 90 s，我们认为此时很多神经元尚未开始发放，导致模型未能充分学习到发放模式；
- **zero-shot 到不同电极会发生灾难性遗忘（catastrophic forgetting）**，因此需要 few-shot 或持续学习（相关实验见 `options_linux_list` 中的 `ewc` / `kd` / `l2` 等变体配置）。

## 存在不足与未来方向

1. **长期记录未验证**：目前主要针对急性记录（acute recordings），长期记录（chronic）下的表现未知；
2. **时序建模不足**：模型设计主要受 SpkRecon 与图像超分辨率思路启发，尽管探索了滑窗的影响，但对神经信号长程时序依赖的建模仍有欠缺。MAE与随机掩码等自监督范式可能更为适配；
3. **鲁棒性验证有限**：理论上，深度学习方法在随机损坏通道上应表现更优、更具鲁棒性，但由于显卡资源限制，目前仅在小规模迁移评估数据集上完成了验证（结果证实了这一点）；
4. **资源受限导致的优化不足**：受限于硬件与预算，许多工程与调优细节尚未充分打磨（Sorry，实在太穷了）；
5. **数据预处理范式值得深挖**：不同模态（ECoG 的 Z-Score 归一化、Spike Rate 的不应期上界约束、图像的像素值缩放、LLM 的分词器离散化）预处理逻辑各异。本项目沿用 ECoG 领域的 Z-Score 方案，但神经科学数据特有的预处理范式及其对模型泛化性的影响仍有待进一步探索。
