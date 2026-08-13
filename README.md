# Spk-Res

**English** | [简体中文](./README_CN.md)

## Introduction

**Spk-Res** is a deep-learning-based framework for restoring neural signals on bad channels of high-density probes.

Signals recorded by high-density electrode arrays leave spatial footprints on neighboring electrodes, giving rise to a spatial relationship between channels that can be exploited for interpolation. Traditionally, Kriging interpolation has been the method of choice for such bad-channel imputation. This project seeks to answer the following questions:

1. **Can Kriging capture the underlying non-linear spatial relationship?** In particular, when the signal of the main channel is missing, can reconstruction be performed in an online manner during recording?
2. **Do deep learning models outperform traditional methods?** We compare Kriging against deep-learning-based reconstruction and evaluate the fidelity of reconstructed neural waveforms on damaged channels.
3. **What do the models actually learn?** We apply **LAM (Local Attribution Map)** to analyze the trained models. By generating local attribution maps via backpropagation (e.g., gradient / path-gradient methods), we quantify how much a model relies on each input region when reconstructing a specific channel or temporal window, and we compare the influence of different blur paths (e.g., linear interpolation vs. Gaussian blur paths) and attribution strategies to uncover the spatial features the model depends on.

## Datasets

| Dataset | Purpose | Source |
| --- | --- | --- |
| Overlapping Datasets | General training / validation | [DANDI 000034](https://dandiarchive.org/dandiset/000034) |
| Drifting Datasets | Zero-shot transfer evaluation under drift | [Steinmetz et al., Science 2021](https://figshare.com/articles/dataset/_Imposed_motion_datasets_from_Steinmetz_et_al_Science_2021/14024495) |
| Transfer Evaluation Dataset | Transfer evaluation (small-scale dataset) | [Zenodo 3696926](https://zenodo.org/records/3696926) |

## Installation

Create a `conda` environment and install the dependencies:

```bash
conda create -n spk_res python=3.9
conda activate spk_res
pip install -r requirements.txt
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

> This project is refactored on top of [BasicSR](https://github.com/XPixelGroup/BasicSR). Core dependencies include PyTorch, SpikeInterface, MEArec, KiloSort, MountainSort5, and other toolkits from the neuroscience and super-resolution ecosystems.

## Project Layout

```
Spk-Res/
├── main_binfile_v2.py      # .bin recordings: data generation → training → reconstruction
├── main_meafile_v2.py      # .h5 (MEArec) recordings: full pipeline + spike-sorter evaluation
├── main_nwbfile_v2.py      # .nwb recordings entry point
├── test_sorter_h5.py       # Runs spike sorters to evaluate reconstruction quality
├── res_utils/              # Core utilities (data generation, training, evaluation, LAM attribution)
├── basicsr/                # BasicSR-based model zoo and training framework
├── options_linux/          # Training / testing configuration files (.yml)
├── options_linux_list/     # Batch experiment configs (incl. continual-learning / distillation variants)
├── analysis/               # Result analysis scripts and notebooks
├── tables/                 # Evaluation tables across methods and sorters (F1 / Precision / Recall)
└── LAM/                    # Local Attribution Map analysis
```

## Getting Started

### 1. Configure the experiment

All parameters are set via YAML files under `options_linux/` and `options_linux_list/`, including:

- **Data generation settings** (`settings`): downsampling factor `factor`, whether to use Kriging rendering, sliding-window size `slide_window`, train/test data ratio, etc.;
- **Network architecture** (`network_g`): EDSR and Restormer backbones are supported;
- **Training settings** (`train`): optimizer, learning-rate schedule, loss function (MSE / L1), etc.;
- **Validation & logging** (`val` / `logger`).

### 2. Run the pipeline

Taking `.h5` recordings as an example (`main_meafile_v2.py`):

```bash
python main_meafile_v2.py
```

The full pipeline consists of four stages:

1. **Generate training data** — mask out bad channels (damaged-channel simulation) on raw recordings and render training samples;
2. **Train the model** — train the reconstruction network on the generated training set;
3. **Reconstruct recordings** — restore the damaged channels and compute metrics such as MSE / NRMSE;
4. **Downstream evaluation** — run spike sorters (e.g., MountainSort5, HerdingSpikes) on the reconstructed signals, then compute hit rate, F1, Precision, and Recall.

> Note: The paths in the scripts are examples. Please update `settings.rec_path`, `save_path`, `weights_path`, etc. in `options_linux/test_bin_demo.yml` according to your actual data location.

## Experiments & Findings

### 1. Overlapping Datasets

- On acute recordings, as little as the first 30% of data is sufficient to reconstruct the remaining 70%;
- Compared with Kriging, deep learning methods reconstruct neural waveforms on damaged channels with higher fidelity;
- The more channels that are damaged, the harder the reconstruction becomes and the higher the error.

### 2. Drifting Datasets

- Applying weights pretrained on other electrodes **directly via zero-shot** to data with the same drift pattern drives MSE very low; however, **a low MSE does not imply good reconstruction quality** — it must be assessed together with the clustering quality of the downstream spike sorter.

### 3. Transfer Evaluation Datasets

- Compared with the first two datasets, this one requires a higher training-data ratio. The main reason is its short duration (300 s): at a 30% training ratio, only ~90 s of data would be used;
- **Zero-shot transfer to different electrodes suffers from catastrophic forgetting**, which motivates few-shot or continual-learning approaches (see the `ewc` / `kd` / `l2` variants under `options_linux_list`).

## Limitations & Future Directions

1. **Chronic recordings not yet validated**: current experiments focus on acute recordings; performance on chronic (long-term) recordings is unknown;
2. **Insufficient temporal modeling**: the model design is mainly inspired by SpkRecon and image super-resolution. Although the influence of the sliding window has been explored, modeling long-range temporal dependencies of neural signals remains an open issue. Self-supervised paradigms such as MAE (masked autoencoding) and random masking may be better suited;
3. **Limited robustness validation**: in theory, deep learning methods should perform better and be more robust on randomly damaged channels. Due to GPU constraints, this has so far only been verified on the small-scale transfer evaluation dataset (and indeed holds there);
4. **Suboptimal engineering due to limited resources**: many engineering details and tuning steps remain unpolished given hardware and budget constraints (and i am stupid);
5. **Preprocessing paradigms deserve further exploration**: different modalities employ different preprocessing logic (Z-Score normalization for ECoG, refractory-period upper bounds for spike rates, pixel scaling for images, tokenizer discretization for LLMs). This project adopts the Z-Score scheme from the ECoG field, but the preprocessing paradigms specific to neuroscience data and their impact on model generalization are still worth investigating.
