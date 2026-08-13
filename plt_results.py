"""
plt_results.py — Spike sorting 结果可视化脚本

命名规范:
  factor_N   — 坏通道比例 1/N (越小越严重)
  np128      — Neuropixels-128 电极; sq100 — SqMEA 电极
  _rd        — random bad channels
  _nm        — 输入经过 Z-Score 归一化 (对 EDSR / SpkRes 深度学习方法有意义)

对比目标:
  1. 哪种重建方法更好?  (Kriging vs Remove vs EDSR, 均不归一化 fair comparison)
  2. Z-Score 归一化是否提升 DL 方法排序精度?  (EDSR & SpkRes w/ Normalization vs w/o Normalization)

指标:
  Recall    = well_detected / num_gt      (召回率 / Sensitivity)
  Precision = well_detected / num_sorter  (精确率 / Purity)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# ============================================================
# 全局样式 — Arial, 8pt, low-saturation
# ============================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 800,
    'savefig.dpi': 800,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'legend.labelspacing': 0.15,
})

# Unified low-saturation palette
CLR = {
    'krig':        '#7E9FC4',   # muted blue
    'remove':      '#D4A87C',   # muted orange
    'edsr':        '#8CB896',   # muted green
    'spkres':      '#BC8F8F',   # muted rose
    'edsr_norm':   '#7BA87B',   # darker muted green (w/ Normalization)
    'spkres_norm': '#9B6B6B',   # darker muted rose (w/ Normalization)
    'baseline':    '#B0B0B0',   # grey baseline
    'transfer':    '#C98F8F',   # muted red (direct_transfer)
    'l2':          '#7E9FC4',   # muted blue
    'llr':         '#D4A87C',   # muted orange
    'ewc':         '#8CB896',   # muted green
    'kd':          '#C5A0C5',   # muted purple
    'erl':         '#A0A0A0',   # grey
    'erl_kd':      '#C5A0C5',   # muted purple
    'fine_tuned':  '#D08C8C',   # muted red
    'vae':         '#D4A87C',   # muted orange
}

METHODS = ['krig', 'remove', 'edsr', 'spkres']
METHOD_COLORS = {m: CLR[m] for m in METHODS}
METHOD_LABELS = {'krig': 'Kriging', 'remove': 'Remove', 'edsr': 'EDSR', 'spkres': 'SpkRes'}
ELECTRODE_LABELS = {'np128': 'Neuropixels-128', 'sq100': 'SqMEA-10-15'}
EDSR_NORM_COLORS = {'w/o Normalization': CLR['edsr'], 'w/ Normalization': CLR['edsr_norm']}
SPKRES_NORM_COLORS = {'w/o Normalization': CLR['spkres'], 'w/ Normalization': CLR['spkres_norm']}

def _get_norm_colors(method):
    """Return norm colors dict for the given method."""
    if method == 'edsr':
        return EDSR_NORM_COLORS
    elif method == 'spkres':
        return SPKRES_NORM_COLORS
    return {'w/o Normalization': CLR.get(method, '#999'), 'w/ Normalization': CLR.get(f'{method}_norm', '#666')}

# ============================================================
# 数据加载
# ============================================================

def discover_all_data(base_dir='sorter'):
    records = []
    model_dirs = sorted([d for d in os.listdir(base_dir)
                         if os.path.isdir(os.path.join(base_dir, d))])
    for model in model_dirs:
        model_path = os.path.join(base_dir, model)
        factor_dirs = sorted([d for d in os.listdir(model_path)
                              if os.path.isdir(os.path.join(model_path, d)) and d.startswith('factor_')])
        for fd in factor_dirs:
            factor = int(fd.split('_')[1])
            factor_path = os.path.join(model_path, fd)
            config_dirs = sorted([d for d in os.listdir(factor_path)
                                  if os.path.isdir(os.path.join(factor_path, d))])
            for cfg in config_dirs:
                csv_path = os.path.join(factor_path, cfg, 'unit_counts.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['factor'] = factor
                    df['config'] = cfg
                    df['model'] = model
                    records.append(df)
    df_all = pd.concat(records, ignore_index=True)
    df_all = df_all.rename(columns={'dataset': 'method'})
    df_all = df_all[~((df_all['model'] == 'spkres') & (df_all['method'] != 'spkres'))]
    df_all['electrode'] = df_all['config'].apply(
        lambda x: 'np128' if ('np128' in x or '_np' in x) else 'sq100' if ('sq100' in x or 'sqmea' in x or '_sq' in x) else 'other')
    df_all['use_norm'] = df_all['config'].apply(
        lambda x: 'w/ Normalization' if x.endswith('_nm') else 'w/o Normalization')
    return df_all


def get_avg_across_sorters(df):
    """对每个 (factor, config, method) 组合，取两种 sorter 的均值。"""
    return df.groupby(['factor', 'config', 'method', 'electrode', 'use_norm'], as_index=False).agg({
        'num_gt': 'first',
        'num_sorter': 'mean',
        'num_well_detected': 'mean',
        'num_false_positive': 'mean',
        'num_redundant': 'mean',
        'num_overmerged': 'mean',
        'num_bad': 'mean',
    })


# ============================================================
# 绘图辅助
# ============================================================

def _get_factors_desc(df):
    """返回降序排列的 factor 列表 (16, 8, 4, 2), 对应标签 1/16→1/2"""
    return sorted(df['factor'].unique(), reverse=True)


def _add_factor_labels(ax, factors, xlabel='Bad Channel Ratio'):
    """设置 X 轴标签: 1/16, 1/8, 1/4, 1/2"""
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([f'1/{f}' for f in factors])
    ax.set_xlabel(xlabel)


def _annotate_gt(ax, gt_val):
    """在 plot 右上角内侧标注 GT 值。"""
    ax.axhline(y=gt_val, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.annotate(f'GT={int(gt_val)}', xy=(0.98, 0.92), xycoords='axes fraction',
                fontsize=8, color='gray', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))


def _compute_metrics(edf):
    """计算 Recall, Precision, F1 百分比列。"""
    edf['recall'] = edf['num_well_detected'] / edf['num_gt'] * 100
    edf['precision'] = edf['num_well_detected'] / edf['num_sorter'] * 100
    edf['f1'] = 2 * (edf['recall'] * edf['precision']) / (edf['recall'] + edf['precision']).replace(0, np.nan)
    return edf


def _collapse_configs(df_avg):
    """当同一 (factor, method, electrode, use_norm) 组合有多条 config 时取均值去重。"""
    return df_avg.groupby(['factor', 'method', 'electrode', 'use_norm'], as_index=False).agg({
        'num_gt': 'first',
        'num_sorter': 'mean',
        'num_well_detected': 'mean',
        'num_false_positive': 'mean',
        'num_redundant': 'mean',
        'num_overmerged': 'mean',
        'num_bad': 'mean',
    })


# ============================================================
# 图 1: 方法对比 — Well-Detected Units (fair comparison, w/o Normalization)
# ============================================================

def plot_method_comparison(df, electrode='np128', save_path=None):
    """
    核心图: 四种重建方法横向对比 (均 w/o Normalization)。
    生成两幅子图: Bar (Well-Detected Units) + F1 (%)。
    同时保存 per-sorter CSV 表格。
    """
    # 平均数据用于绘图
    df_avg = get_avg_across_sorters(df)
    edf_all_cfg = df_avg[(df_avg['electrode'] == electrode) & (df_avg['use_norm'] == 'w/o Normalization')].copy()
    edf_all_cfg = _compute_metrics(edf_all_cfg)
    factors = _get_factors_desc(edf_all_cfg)

    # per-sorter 原始数据用于表格
    edf_raw = df[(df['electrode'] == electrode) &
                 (df['use_norm'] == 'w/o Normalization')].copy()
    edf_raw = _compute_per_sorter_metrics(edf_raw)

    if save_path:
        base = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'tiff'
    
    # without normalization  
    if electrode == 'np128':
        cfg_name = 'single_np_rd'
    elif electrode == 'sq100':
        cfg_name = 'single_sq_rd'
    edf = edf_all_cfg[edf_all_cfg['config'] == cfg_name]
    # ---- 子图1: Well-Detected Units 柱状图 ----
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    x = np.arange(len(factors))
    bar_width = 0.2

    for mi, method in enumerate(METHODS):
        subset = edf[edf['method'] == method].set_index('factor').reindex(factors).reset_index()
        offset = (mi - 1.5) * bar_width
        ax.bar(x + offset, subset['num_well_detected'].values, bar_width,
               color=METHOD_COLORS[method], edgecolor='black', linewidth=0.5,
               label=METHOD_LABELS[method])

    gt_val = edf['num_gt'].iloc[0] if len(edf) > 0 else 0
    _annotate_gt(ax, gt_val)
    _add_factor_labels(ax, factors)
    ax.set_ylabel('Well-Detected Units')
    ax.set_title(f'Method Comparison — {ELECTRODE_LABELS.get(electrode, electrode)}', fontsize=8)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False)
    plt.tight_layout()
    plt.show()
    if save_path:
        fig.savefig(f'{base}_bar.{ext}', bbox_inches='tight')
        print(f'[Saved] {base}_bar.{ext}')
    plt.close(fig)

    # ---- 子图2: F1 Score 折线图 ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
    for method in METHODS:
        ndf = edf[edf['method'] == method].set_index('factor').reindex(factors).reset_index()
        if len(ndf) == 0:
            continue
        pos = [factors.index(f) for f in ndf['factor']]
        ax2.plot(pos, ndf['f1'], marker='o', linestyle='-',
                color=METHOD_COLORS[method], linewidth=1, markersize=2,
                label=METHOD_LABELS[method])

    _add_factor_labels(ax2, factors)
    ax2.set_ylabel('F1 (%)')
    ax2.set_ylim(0, 105)
    ax2.set_title(f'F1 Score — {ELECTRODE_LABELS.get(electrode, electrode)}', fontsize=8)
    ax2.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0.02, 0.02), frameon=False)
    plt.tight_layout()
    if save_path:
        fig2.savefig(f'{base}_f1.{ext}', bbox_inches='tight')
        print(f'[Saved] {base}_f1.{ext}')
    plt.close(fig2)

    # ---- 导出 per-sorter CSV 表格 ----
    generate_method_comparison_tables(edf_raw, factors, electrode)


def _compute_per_sorter_metrics(edf):
    """对 per-sorter 原始数据计算 Recall, Precision, F1 (小数形式)。"""
    edf['recall'] = edf['num_well_detected'] / edf['num_gt']
    edf['precision'] = edf['num_well_detected'] / edf['num_sorter']
    edf['f1'] = 2 * (edf['recall'] * edf['precision']) / (edf['recall'] + edf['precision']).replace(0, np.nan)
    return edf


def generate_method_comparison_tables(edf_raw, factors, electrode):
    """将 method comparison 的 per-sorter 数据导出为方法对比 CSV 和 LaTeX 三线表。"""
    os.makedirs(f'tables/{electrode}', exist_ok=True)
    # os.makedirs(f'tables_latex/{electrode}', exist_ok=True)

    # 指标列表: (内部列名, 显示名称, 小数位数)
    metrics_list = ['factor', 'method', 'num_sorter', 'num_well_detected', 'recall', 'precision', 'f1']
    num_list = ['factor', 'num_sorter', 'num_well_detected']
    other_list = ['recall', 'precision', 'f1']

    # ---- 构建 pivot 表格: Factor | Method | Detected | Well-Detected | False Positive | Redundant | Overmerged | Bad | Recall | Precision | F1 ----
    config_list = edf_raw['config'].dropna().unique()
    for cfg_i in config_list:
        df_cfg = edf_raw[edf_raw['config'] == cfg_i]  #  single_np_rd / ...
        for sorter_name in sorted(df_cfg['sorter_name'].dropna().unique()):
            rows = []
            for factor in factors:
                for method in METHODS:
                    sub = df_cfg[(df_cfg['factor'] == factor) &
                                (df_cfg['method'] == method) &
                                (df_cfg['sorter_name'] == sorter_name)]
                    rows.append(sub)
                
            df_out = pd.concat(rows, ignore_index=True)[metrics_list]
            df_out[num_list] = df_out[num_list].astype(int)
            df_out[other_list] = df_out[other_list].round(3)

            csv_path = f'tables/{electrode}/{cfg_i}_{sorter_name}.csv'
            df_out.to_csv(csv_path, index=False)
            print(f'[Table Saved] {csv_path}')

            # ---- LaTeX 三线表 ----
            # generate_method_comparison_latex(df_out, electrode, cfg_i, sorter_name)

            gt_val = edf_raw['num_gt'].iloc[0] if len(edf_raw) > 0 else 0
            print(f'[Info] {ELECTRODE_LABELS.get(electrode, electrode)} GT = {int(gt_val)}')

# ============================================================
# 图 2: Normalization 效果 (多模型并排对比)
# ============================================================

def plot_norm_effect(df, models=None, electrode='np128', save_path=None):
    """
    Normalization 对 DL 方法的效果图 (拆分为 4 个独立子图)。
    默认 models=['edsr', 'spkres'], 所有模型并行展示。
    """
    if models is None:
        models = ['edsr', 'spkres']

    df_avg = get_avg_across_sorters(df)
    all_factors = []
    all_gt = []
    dfs_by_model = {}

    for model in models:
        edf = df_avg[(df_avg['electrode'] == electrode) & (df_avg['method'] == model)].copy()
        edf = _compute_metrics(edf)
        dfs_by_model[model] = edf
        if len(edf) > 0:
            all_factors.append(edf['factor'].unique())
            all_gt.append(edf['num_gt'].iloc[0])

    factors = sorted(set().union(*all_factors), reverse=True) if all_factors else []
    gt_val = all_gt[0] if all_gt else 0

    norm_conditions = ['w/o Normalization', 'w/ Normalization']
    norm_hatches = {'w/o Normalization': '', 'w/ Normalization': '//'}

    combo_configs = {}
    for model in models:
        nc = _get_norm_colors(model)
        label = METHOD_LABELS.get(model, model.upper())
        for nm in norm_conditions:
            combo_configs[(model, nm)] = {'color': nc[nm], 'label': f'{label} {nm}'}

    MODEL_MARKERS = {'edsr': {'w/o Normalization': 'o', 'w/ Normalization': 's'},
                     'spkres': {'w/o Normalization': 'D', 'w/ Normalization': '^'}}

    if save_path:
        base = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'tiff'

    # ---- well-detected 柱状图 ----
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
    n_groups = len(models) * 2
    bar_width = 0.8 / n_groups
    combo_list = [(m, nm) for m in models for nm in norm_conditions]
    x = np.arange(len(factors))

    for ci, (model, nm) in enumerate(combo_list):

        cfg_name = None
        edf = dfs_by_model.get(model)
        if electrode == 'np128' and nm == 'w/o Normalization':
            cfg_name = 'single_np_rd'
        elif electrode == 'np128' and nm == 'w/ Normalization':
            cfg_name = 'single_np_rd_nm'
        elif electrode == 'sq100' and nm == 'w/o Normalization':
            cfg_name = 'single_sq_rd'
        elif electrode == 'sq100' and nm == 'w/ Normalization':
            cfg_name = 'single_sq_rd_nm'
        edf_cfg = edf[edf['config'] == cfg_name]
        
        ndf = edf_cfg[edf_cfg['use_norm'] == nm].set_index('factor').reindex(factors).reset_index()
        offset = (ci - (n_groups - 1) / 2) * bar_width
        cfg = combo_configs[(model, nm)]
        ax1.bar(x + offset, ndf['num_well_detected'].values, bar_width,
                color=cfg['color'], hatch=norm_hatches.get(nm, ''),
                edgecolor='black', linewidth=0.5, label=cfg['label'])

    _annotate_gt(ax1, gt_val)
    _add_factor_labels(ax1, factors)
    ax1.set_ylabel('Well-Detected Units')
    ax1.set_title(f'Well-Detected — {ELECTRODE_LABELS.get(electrode, electrode)}', fontsize=8)
    ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 0.99), frameon=False)
    plt.tight_layout()
    if save_path:
        fig1.savefig(f'{base}_bar.{ext}', bbox_inches='tight')
        print(f'[Saved] {base}_bar.{ext}')
    plt.close(fig1)

    # ---- Recall / Precision / F1 折线图 ----
    # for metric, ylbl in [('recall', 'Recall'), ('precision', 'Precision'), ('f1', 'F1')]:
    for metric, ylbl in [('f1', 'F1')]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        for model in models:
            for nm in norm_conditions:

                cfg_name = None
                edf = dfs_by_model.get(model)
                if electrode == 'np128' and nm == 'w/o Normalization':
                    cfg_name = 'single_np_rd'
                elif electrode == 'np128' and nm == 'w/ Normalization':
                    cfg_name = 'single_np_rd_nm'
                elif electrode == 'sq100' and nm == 'w/o Normalization':
                    cfg_name = 'single_sq_rd'
                elif electrode == 'sq100' and nm == 'w/ Normalization':
                    cfg_name = 'single_sq_rd_nm'
                edf_cfg = edf[edf['config'] == cfg_name]

                ndf = edf_cfg[edf_cfg['use_norm'] == nm].set_index('factor').reindex(factors).reset_index()
                pos = [factors.index(f) for f in ndf['factor']]
                cfg = combo_configs[(model, nm)]
                ax.plot(pos, ndf[metric], marker=MODEL_MARKERS[model][nm], linestyle='-',
                        linewidth=1, markersize=2, color=cfg['color'], label=cfg['label'])
        _add_factor_labels(ax, factors)
        ax.set_ylabel(f'{ylbl} (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'{ylbl} — {ELECTRODE_LABELS.get(electrode, electrode)}', fontsize=8)
        # ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False)
        ax.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0.02, 0.02), frameon=False)
        plt.tight_layout()
        if save_path:
            fig.savefig(f'{base}_{metric}.{ext}', bbox_inches='tight')
            print(f'[Saved] {base}_{metric}.{ext}')
        plt.close(fig)


def plot_norm_effect_edsr(df, electrode='np128', save_path=None):
    """Backward-compatible wrapper for EDSR normalization effect."""
    return plot_norm_effect(df, models=['edsr'], electrode=electrode, save_path=save_path)


# ============================================================
# 图 3: Recall & Precision 对比
# ============================================================

def plot_efficiency_comparison(df, electrode='np128', save_path=None):
    """
    Recall / Precision / F1 独立成图, 对比归一化对各指标的影响。
    包含 Kriging / Remove / EDSR(w/o & w/ Norm) / SpkRes(w/o & w/ Norm)
    """
    df_avg = get_avg_across_sorters(df)
    edf = df_avg[df_avg['electrode'] == electrode].copy()
    edf = _compute_metrics(edf)
    factors = _get_factors_desc(edf)
    spkres_nc = SPKRES_NORM_COLORS

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=METHOD_COLORS['krig'], marker='o', linestyle='-',
               linewidth=1, markersize=2, label=METHOD_LABELS['krig']),
        Line2D([0], [0], color=METHOD_COLORS['remove'], marker='o', linestyle='-',
               linewidth=1, markersize=2, label=METHOD_LABELS['remove']),
        Line2D([0], [0], color=EDSR_NORM_COLORS['w/o Normalization'], marker='o', linestyle='-',
               linewidth=1, markersize=2, label='EDSR w/o Norm.'),
        Line2D([0], [0], color=EDSR_NORM_COLORS['w/ Normalization'], marker='s', linestyle='--',
               linewidth=1, markersize=2, label='EDSR w/ Norm.'),
        Line2D([0], [0], color=spkres_nc['w/o Normalization'], marker='D', linestyle='-',
               linewidth=1, markersize=2, label='SpkRes w/o Norm.'),
        Line2D([0], [0], color=spkres_nc['w/ Normalization'], marker='^', linestyle='--',
               linewidth=1, markersize=2, label='SpkRes w/ Norm.'),
    ]

    if save_path:
        base = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'tiff'

    for metric_col, metric_label in [('f1', 'F1')]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        for method in ['krig', 'remove']:
            ndf = edf[(edf['method'] == method) & (edf['use_norm'] == 'w/o Normalization')] \
                  .set_index('factor').reindex(factors).reset_index()
            if len(ndf) == 0:
                continue
            pos = [factors.index(f) for f in ndf['factor']]
            ax.plot(pos, ndf[metric_col], 'o-', color=METHOD_COLORS[method], linewidth=1, markersize=2)

        for ni, nm in enumerate(['w/o Normalization', 'w/ Normalization']):
            ndf = edf[(edf['method'] == 'edsr') & (edf['use_norm'] == nm)] \
                  .set_index('factor').reindex(factors).reset_index()
            if len(ndf) > 0:
                pos = [factors.index(f) for f in ndf['factor']]
                ax.plot(pos, ndf[metric_col], marker='o' if ni == 0 else 's',
                        linestyle='-' if ni == 0 else '--', color=EDSR_NORM_COLORS[nm],
                        linewidth=1, markersize=2)

        for ni, nm in enumerate(['w/o Normalization', 'w/ Normalization']):
            ndf = edf[(edf['method'] == 'spkres') & (edf['use_norm'] == nm)] \
                  .set_index('factor').reindex(factors).reset_index()
            if len(ndf) > 0:
                pos = [factors.index(f) for f in ndf['factor']]
                ax.plot(pos, ndf[metric_col], marker='D' if ni == 0 else '^',
                        linestyle='-' if ni == 0 else '--', color=spkres_nc[nm],
                        linewidth=1, markersize=2)

        _add_factor_labels(ax, factors)
        ax.set_ylabel(f'{metric_label} (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'{metric_label} — {ELECTRODE_LABELS.get(electrode, electrode)}', fontsize=8)
        ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.02, 0.02),
                  ncol=1, fontsize=8, frameon=False)
        plt.tight_layout()
        if save_path:
            fig.savefig(f'{base}_{metric_col}.{ext}', bbox_inches='tight')
            print(f'[Saved] {base}_{metric_col}.{ext}')
        plt.close(fig)

# ============================================================
# 图 5: Zero-shot vs Few-shot 迁移对比 (sq100 目标域)
# ============================================================

def plot_zero_shot_vs_few_shot(df_all, save_path=None):
    """
    SqMEA 迁移学习对比: Kriging / Remove / Zero-shot / Few-shot (EDSR & SpkRes)。
    """
    records = []
    pairings = [
        # Baselines
        ('single_sq_rd_nm', 'krig', 'Kriging'),
        ('single_sq_rd_nm', 'remove', 'Remove'),
        # EDSR transfer
        ('zs_sq_rd_nm', 'edsr', 'Zero-shot (EDSR)'),
        ('zs_sq_rd_nm_gcl', 'edsr', 'Zero-shot GCL (EDSR)'),
        ('fs_sq_rd_nm_p1_5', 'edsr', 'Few-shot (EDSR)'),
        # SpkRes transfer
        ('zs_sq_rd_nm', 'spkres', 'Zero-shot (SpkRes)'),
        ('zs_sq_rd_nm_gcl', 'spkres', 'Zero-shot GCL (SpkRes)'),
        ('fs_sq_rd_nm_p1_5', 'spkres', 'Few-shot (SpkRes)'),
    ]

    for cfg, method, label in pairings:
        sub = df_all[(df_all['config'] == cfg) & (df_all['method'] == method)].copy()
        sub['display'] = label
        records.append(sub)
    edf_raw = pd.concat(records, ignore_index=True)
    df_avg = edf_raw.groupby(['factor', 'display', 'electrode', 'use_norm'], as_index=False).agg({
        'num_gt': 'first', 'num_sorter': 'mean', 'num_well_detected': 'mean',
        'num_false_positive': 'mean', 'num_redundant': 'mean',
        'num_overmerged': 'mean', 'num_bad': 'mean',
    })
    edf = _compute_metrics(df_avg)

    displays = [p[2] for p in pairings]
    display_colors = {
        'Kriging':              CLR['krig'],
        'Remove':               CLR['remove'],
        'Zero-shot (EDSR)':     '#A0C8E0',
        'Zero-shot GCL (EDSR)': '#7EC8B8',
        'Few-shot (EDSR)':      CLR['edsr'],
        'Zero-shot (SpkRes)':   '#D4A0A0',
        'Zero-shot GCL (SpkRes)': '#C08080',
        'Few-shot (SpkRes)':    CLR['spkres'],
    }
    display_markers = {
        'Kriging':              'o', 'Remove': 's',
        'Zero-shot (EDSR)':            'D', 'Zero-shot GCL (EDSR)': 'v', 'Few-shot (EDSR)': '^',
        'Zero-shot (SpkRes)':   '<', 'Zero-shot GCL (SpkRes)': '>', 'Few-shot (SpkRes)': 'P',
    }
    is_transfer = {
        'Kriging': False, 'Remove': False,
        'Zero-shot (EDSR)': True, 'Zero-shot GCL (EDSR)': True, 'Few-shot (EDSR)': True,
        'Zero-shot (SpkRes)': True, 'Zero-shot GCL (SpkRes)': True, 'Few-shot (SpkRes)': True,
    }
    edsr_displays = ['Zero-shot (EDSR)', 'Zero-shot GCL (EDSR)', 'Few-shot (EDSR)']
    spkres_displays = ['Zero-shot (SpkRes)', 'Zero-shot GCL (SpkRes)', 'Few-shot (SpkRes)']

    factors = _get_factors_desc(edf)
    gt_val = edf['num_gt'].iloc[0] if len(edf) > 0 else 0
    n = len(displays)
    bar_w = 0.8 / n

    if save_path:
        base = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'tiff'

    # --- Fig 5a: Well-Detected grouped bar ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
    x = np.arange(len(factors))
    for ci, d in enumerate(displays):
        ndf = edf[edf['display'] == d].set_index('factor').reindex(factors).reset_index()
        if len(ndf) == 0:
            continue
        offset = (ci - (n - 1) / 2) * bar_w
        ax1.bar(x + offset, ndf['num_well_detected'].values, bar_w,
                color=display_colors[d], edgecolor='black', linewidth=0.3, label=d)
    _annotate_gt(ax1, gt_val)
    _add_factor_labels(ax1, factors)
    ax1.set_ylabel('Well-Detected Units')
    ax1.set_title('Well-Detected (SqMEA)', fontsize=8)
    ax1.legend(fontsize=8, loc='upper left', frameon=False, bbox_to_anchor=(0, 0.98))
    plt.tight_layout()
    if save_path:
        fig1.savefig(f'{base}_bar.{ext}', bbox_inches='tight')
        print(f'[Saved] {base}_bar.{ext}')
    plt.close(fig1)

    # --- Fig 5b–5d: Recall / Precision / F1 所有方法 ---
    for metric_col, metric_label in [('recall', 'Recall'), ('precision', 'Precision'), ('f1', 'F1')]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for d in displays:
            ndf = edf[edf['display'] == d].set_index('factor').reindex(factors).reset_index()
            if len(ndf) == 0:
                continue
            pos = [factors.index(f) for f in ndf['factor']]
            ax.plot(pos, ndf[metric_col], marker=display_markers[d],
                    linestyle='--' if not is_transfer[d] else '-',
                    color=display_colors[d], linewidth=1, markersize=2, label=d)
        _add_factor_labels(ax, factors)
        ax.set_ylabel(f'{metric_label} (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'{metric_label} (SqMEA)', fontsize=8)
        ax.legend(fontsize=8, loc='upper left', frameon=False,ncol=2)
        plt.tight_layout()
        if save_path:
            fig.savefig(f'{base}_{metric_col}.{ext}', bbox_inches='tight')
            print(f'[Saved] {base}_{metric_col}.{ext}')
        plt.close(fig)

    # --- Fig 5e–5g: EDSR Transfer zoom ---
    for metric_col, metric_label in [('f1', 'F1')]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for d in edsr_displays:
            ndf = edf[edf['display'] == d].set_index('factor').reindex(factors).reset_index()
            if len(ndf) == 0:
                continue
            pos = [factors.index(f) for f in ndf['factor']]
            ax.plot(pos, ndf[metric_col], marker=display_markers[d], linestyle='-',
                    color=display_colors[d], linewidth=1, markersize=2, label=d)
        _add_factor_labels(ax, factors)
        ax.set_ylabel(f'{metric_label} (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'EDSR Transfer: {metric_label}', fontsize=8)
        ax.legend(fontsize=8, loc='upper left', frameon=False)
        plt.tight_layout()
        if save_path:
            fig.savefig(f'{base}_edsr_{metric_col}.{ext}', bbox_inches='tight')
            print(f'[Saved] {base}_edsr_{metric_col}.{ext}')
        plt.close(fig)

    # --- Fig 5h–5j: SpkRes Transfer zoom ---
    for metric_col, metric_label in [('f1', 'F1')]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for d in spkres_displays:
            ndf = edf[edf['display'] == d].set_index('factor').reindex(factors).reset_index()
            if len(ndf) == 0:
                continue
            pos = [factors.index(f) for f in ndf['factor']]
            ax.plot(pos, ndf[metric_col], marker=display_markers[d], linestyle='-',
                    color=display_colors[d], linewidth=1, markersize=2, label=d)
        _add_factor_labels(ax, factors)
        ax.set_ylabel(f'{metric_label} (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'SpkRes Transfer: {metric_label}', fontsize=8)
        ax.legend(fontsize=8, loc='upper left', frameon=False)
        plt.tight_layout()
        if save_path:
            fig.savefig(f'{base}_spkres_{metric_col}.{ext}', bbox_inches='tight')
            print(f'[Saved] {base}_spkres_{metric_col}.{ext}')
        plt.close(fig)

    # --- Fig 5k: Summary ---
    fig8, ax8 = plt.subplots(1, 1, figsize=(4, 3))
    ax8.axis('off')
    y_pos = 0.98
    ax8.text(0.05, y_pos, 'SqMEA Transfer Learning', fontsize=8, fontweight='bold', va='top', color='#333333')
    y = y_pos - 0.10
    ax8.text(0.05, y, 'Mean across factors:', fontsize=8, fontweight='bold', va='top')
    y -= 0.05
    for d in displays:
        vals_r = edf[edf['display'] == d]['recall']
        vals_p = edf[edf['display'] == d]['precision']
        vals_f1 = edf[edf['display'] == d]['f1']
        if len(vals_r) > 0:
            tag = '(non-learned)' if not is_transfer[d] else '(transfer)'
            ax8.text(0.05, y, f'{d} {tag}:', fontsize=8, va='top', fontweight='bold', color=display_colors[d])
            y -= 0.035
            ax8.text(0.08, y, f'  R={vals_r.mean():.1f}%  P={vals_p.mean():.1f}%  F1={vals_f1.mean():.1f}%',
                     fontsize=8, va='top', color='#555555')
            y -= 0.04
    plt.tight_layout()
    if save_path:
        fig8.savefig(f'{base}_summary.{ext}', bbox_inches='tight')
        print(f'[Saved] {base}_summary.{ext}')
    plt.close(fig8)


# ============================================================
# 图 6: 灾难性遗忘 — 微调后在 np128 源域上的表现
# ============================================================

def plot_catastrophic_forgetting(df_all, save_path=None):
    """
    灾难性遗忘分析 — 拆分为 5 个独立子图，每个 (4, 3)。
    对比 np128 源域上 Kriging / Remove / EDSR / SpkRes 的 fine-tune 前后表现。
    """
    records = []
    pairings = [
        ('single_np_rd_nm', 'krig', 'Kriging'),
        ('single_np_rd_nm', 'remove', 'Remove'),
        ('single_np_rd_nm', 'edsr', 'EDSR'),
        ('rec_np_rd_nm_p1_5', 'edsr', 'EDSR (Fine-tuned)'),
        # SpkRes
        ('single_np_rd_nm', 'spkres', 'SpkRes'),
        ('rec_np_rd_nm_p1_5', 'spkres', 'SpkRes (Fine-tuned)'),
    ]
    for cfg, method, label in pairings:
        sub = df_all[(df_all['config'] == cfg) & (df_all['method'] == method)].copy()
        sub['display'] = label
        records.append(sub)
    edf_raw = pd.concat(records, ignore_index=True)
    df_avg = edf_raw.groupby(['factor', 'display'], as_index=False).agg({
        'num_gt': 'first', 'num_sorter': 'mean', 'num_well_detected': 'mean',
        'num_false_positive': 'mean', 'num_redundant': 'mean',
        'num_overmerged': 'mean', 'num_bad': 'mean',
    })
    edf = _compute_metrics(df_avg)

    displays = [p[2] for p in pairings]
    display_colors = {
        'Kriging':               CLR['krig'],
        'Remove':                CLR['remove'],
        'EDSR':                  CLR['edsr'],
        'EDSR (Fine-tuned)':     CLR['fine_tuned'],
        'SpkRes':                CLR['spkres'],
        'SpkRes (Fine-tuned)':   CLR['spkres_norm'],
    }
    display_markers = {
        'Kriging': 'o', 'Remove': 's',
        'EDSR': 'D', 'EDSR (Fine-tuned)': '^',
        'SpkRes': 'P', 'SpkRes (Fine-tuned)': 'X',
    }
    is_fine_tuned = {
        'Kriging': False, 'Remove': False,
        'EDSR': False, 'EDSR (Fine-tuned)': True,
        'SpkRes': False, 'SpkRes (Fine-tuned)': True,
    }

    factors = _get_factors_desc(edf)
    gt_val = edf['num_gt'].iloc[0] if len(edf) > 0 else 0
    x = np.arange(len(factors))
    n = len(displays)
    bar_w = 0.8 / n

    if save_path:
        base = save_path.rsplit('.', 1)[0]
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'tiff'

    # --- Fig 6a: Well-Detected ---
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
    for ci, d in enumerate(displays):
        ndf = edf[edf['display'] == d].set_index('factor').reindex(factors).reset_index()
        if len(ndf) == 0:
            continue
        offset = (ci - (n - 1) / 2) * bar_w
        hatch = '///' if is_fine_tuned[d] else ''
        ax1.bar(x + offset, ndf['num_well_detected'].values, bar_w,
                color=display_colors[d], edgecolor='black', linewidth=0.3,
                hatch=hatch, label=d)
    _annotate_gt(ax1, gt_val)
    _add_factor_labels(ax1, factors)
    ax1.set_ylabel('Well-Detected Units')
    ax1.set_title('Well-Detected (NP128)', fontsize=8)
    ax1.legend(fontsize=8, loc='upper left', frameon=False, labelspacing=0.15, bbox_to_anchor=(0, 0.99), ncol=2)
    plt.tight_layout()
    if save_path:
        fig1.savefig(f'{base}_bar.{ext}', bbox_inches='tight')
        print(f'[Saved] {base}_bar.{ext}')
    plt.close(fig1)

    # --- Fig 6b–6d: Recall / Precision / F1 ---
    for metric, ylabel in [('recall', 'Recall'), ('precision', 'Precision'), ('f1', 'F1')]:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for d in displays:
            ndf = edf[edf['display'] == d].set_index('factor').reindex(factors).reset_index()
            if len(ndf) == 0:
                continue
            pos = [factors.index(f) for f in ndf['factor']]
            ax.plot(pos, ndf[metric], marker=display_markers[d],
                    linestyle='--' if is_fine_tuned[d] else '-',
                    color=display_colors[d], linewidth=1, markersize=2, label=d)
        _add_factor_labels(ax, factors)
        ax.set_ylabel(f'{ylabel} (%)')
        ax.set_ylim(0, 105)
        ax.set_title(f'{ylabel} (NP128)', fontsize=8)
        ax.legend(fontsize=8, loc='lower left', frameon=False)
        plt.tight_layout()
        if save_path:
            fig.savefig(f'{base}_{metric}.{ext}', bbox_inches='tight')
            print(f'[Saved] {base}_{metric}.{ext}')
        plt.close(fig)

    # --- Fig 6e: Summary + Δ (EDSR & SpkRes) ---
    fig5, ax5 = plt.subplots(1, 1, figsize=(4, 3))
    ax5.axis('off')

    ndf_edsr = edf[edf['display'] == 'EDSR'].set_index('factor')
    ndf_ft = edf[edf['display'] == 'EDSR (Fine-tuned)'].set_index('factor')
    ndf_spk = edf[edf['display'] == 'SpkRes'].set_index('factor')
    ndf_spk_ft = edf[edf['display'] == 'SpkRes (Fine-tuned)'].set_index('factor')

    y_pos = 0.98
    ax5.text(0.05, y_pos, 'Catastrophic Forgetting', fontsize=8, fontweight='bold',
             va='top', color=CLR['fine_tuned'])
    ax5.text(0.05, y_pos - 0.05, 'NP128 Source Domain', fontsize=8, va='top', color='gray')

    # EDSR Δ
    y = y_pos - 0.12
    ax5.text(0.05, y, 'EDSR Δ (Before − Fine-tuned):', fontsize=8, fontweight='bold', va='top')
    y -= 0.04
    for f in factors:
        if f in ndf_edsr.index and f in ndf_ft.index:
            delta_r = ndf_edsr.loc[f, 'recall'] - ndf_ft.loc[f, 'recall']
            delta_p = ndf_edsr.loc[f, 'precision'] - ndf_ft.loc[f, 'precision']
            delta_f1 = ndf_edsr.loc[f, 'f1'] - ndf_ft.loc[f, 'f1']
            delta_wd = ndf_edsr.loc[f, 'num_well_detected'] - ndf_ft.loc[f, 'num_well_detected']
            ax5.text(0.05, y, f'Bad 1/{f}: ΔWD={delta_wd:+.1f}  ΔR={delta_r:+.1f}%  ΔP={delta_p:+.1f}%  ΔF1={delta_f1:+.1f}%',
                     fontsize=7.5, va='top', color=CLR['fine_tuned'] if abs(delta_r) > 1 else '#333333')
            y -= 0.032

    # SpkRes Δ
    y -= 0.01
    ax5.text(0.05, y, 'SpkRes Δ (Before − Fine-tuned):', fontsize=8, fontweight='bold', va='top')
    y -= 0.04
    for f in factors:
        if f in ndf_spk.index and f in ndf_spk_ft.index:
            delta_r = ndf_spk.loc[f, 'recall'] - ndf_spk_ft.loc[f, 'recall']
            delta_p = ndf_spk.loc[f, 'precision'] - ndf_spk_ft.loc[f, 'precision']
            delta_f1 = ndf_spk.loc[f, 'f1'] - ndf_spk_ft.loc[f, 'f1']
            delta_wd = ndf_spk.loc[f, 'num_well_detected'] - ndf_spk_ft.loc[f, 'num_well_detected']
            ax5.text(0.05, y, f'Bad 1/{f}: ΔWD={delta_wd:+.1f}  ΔR={delta_r:+.1f}%  ΔP={delta_p:+.1f}%  ΔF1={delta_f1:+.1f}%',
                     fontsize=7.5, va='top', color=CLR['spkres_norm'] if abs(delta_r) > 1 else '#333333')
            y -= 0.032

    # Mean summary
    y -= 0.02
    ax5.text(0.05, y, 'Mean across factors:', fontsize=8, fontweight='bold', va='top')
    y -= 0.04
    for d in displays:
        vals_r = edf[edf['display'] == d]['recall']
        vals_p = edf[edf['display'] == d]['precision']
        vals_f1 = edf[edf['display'] == d]['f1']
        if len(vals_r) > 0:
            tag = '(DL)' if d in ('EDSR', 'EDSR (Fine-tuned)', 'SpkRes', 'SpkRes (Fine-tuned)') else ''
            ax5.text(0.05, y, f'{d}: R={vals_r.mean():.1f}%  P={vals_p.mean():.1f}%  F1={vals_f1.mean():.1f}% {tag}',
                     fontsize=8, va='top', fontweight='bold', color=display_colors[d])
            y -= 0.04

    y -= 0.02
    ax5.text(0.05, y, 'Kriging/Remove are not learning', fontsize=8, va='top', color='gray', style='italic')
    y -= 0.03
    ax5.text(0.05, y, 'methods → unaffected by fine-tune.', fontsize=8, va='top', color='gray', style='italic')
    plt.tight_layout()
    if save_path:
        fig5.savefig(f'{base}_summary.{ext}', bbox_inches='tight')
        print(f'[Saved] {base}_summary.{ext}')
    plt.close(fig5)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Loading spike sorting results...")
    print("=" * 60)

    df_all = discover_all_data('sorter')
    df_rd = df_all[df_all['config'].str.contains('_rd')].copy()
    print(f"Found {len(df_rd)} data rows")
    print(f"Factors: {sorted(df_rd['factor'].unique())}")
    print(f"Configs: {list(df_rd['config'].unique())}")
    print()

    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)

    for electrode in ['np128', 'sq100']:
        edf = df_rd[df_rd['electrode'] == electrode]
        if len(edf) == 0:
            continue
        print(f"\n--- {ELECTRODE_LABELS[electrode]} ---")

        # 图1: 四种方法 fair comparison (均 w/o Normalization)
        plot_method_comparison(df_rd, electrode=electrode,
            save_path=os.path.join(output_dir, f'1_method_comparison_{electrode}.tiff'))

        # 图2: Normalization 对 EDSR & SpkRes 的效果 (并排对比)
        plot_norm_effect(df_rd, electrode=electrode,
            save_path=os.path.join(output_dir, f'2_norm_effect_{electrode}.tiff'))

        # # 图3: Recall & Precision (含 EDSR & SpkRes norm)
        # plot_efficiency_comparison(df_rd, electrode=electrode,
        #     save_path=os.path.join(output_dir, f'3_recall_precision_{electrode}.png'))

    # 迁移学习分析
    print("\n--- Zero-shot vs Few-shot Transfer ---")
    plot_zero_shot_vs_few_shot(df_all, save_path=os.path.join(output_dir, '5_zero_shot_vs_few_shot.tiff'))

    print("\n--- Catastrophic Forgetting ---")
    plot_catastrophic_forgetting(df_all, save_path=os.path.join(output_dir, '6_catastrophic_forgetting.tiff'))

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()