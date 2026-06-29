#!/usr/bin/env python3
"""
Compare sorting results across continual learning methods for EDSR and SpkRes.

Output: tables and figures saved under sorter/tables/ and sorter/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
})

SORTER_DIR = os.path.join(os.path.dirname(__file__), 'sorter')
TABLE_DIR = 'tables'
FIG_DIR = 'figures'
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FACTORS = ['factor_16', 'factor_8', 'factor_4', 'factor_2']
FACTOR_DISPLAY = ['1/16', '1/8', '1/4', '1/2']
METRICS = ['recall', 'precision', 'f1']

# Models to analyze: edsr and spkres
MODELS = ['edsr', 'spkres']

# Groups: (label, baseline_dir, methods_dict)
# SpkRes uses same config names as EDSR, just under sorter/spkres/
GROUPS = [
    ('np128', 'single_np_rd_nm', {
        'direct_transfer': ['rec_np_rd_nm_p1_5'],
        'l2': ['rec_np_rd_nm_p1_5_l2'],
        'llr': ['rec_np_rd_nm_p1_5_llr'],
        'ewc': ['rec_np_rd_nm_p1_5_ewc'],
        'kd': ['rec_np_rd_nm_p1_5_kd'],
    }),
    ('sqmea', 'single_sq_rd_nm', {
        'direct_transfer': ['fs_sq_rd_nm_p1_5'],
        'l2': ['fs_sq_rd_nm_p1_5_l2'],
        'llr': ['fs_sq_rd_nm_p1_5_llr'],
        'ewc': ['fs_sq_rd_nm_p1_5_ewc'],
        'kd': ['fs_sq_rd_nm_p1_5_kd'],
    }),
]

METHOD_LABELS = {
    'baseline': 'baseline',
    'direct_transfer': 'reverse transfer',
    'l2': 'L2',
    'llr': 'LLR',
    'ewc': 'EWC',
    'kd': 'KD',
}

# Colors: [EDSR, SpkRes] for each method
METHOD_COLORS = [
    ['#B0B0B0', '#C8A0A0'],    # baseline — grey / muted rose-grey
    ['#C98F8F', '#D4A0A0'],    # direct_transfer — muted red / rose
    ['#7E9FC4', '#A0B8D8'],    # l2 — muted blue / light blue
    ['#D4A87C', '#C8986E'],    # llr — muted orange / darker orange
    ['#8CB896', '#A0C8A0'],    # ewc — muted green / light green
    ['#C5A0C5', '#D8B0D8'],    # kd — muted purple / light purple
]


MODEL_LABEL = {'edsr': 'EDSR', 'spkres': 'SpkRes'}
MODEL_HATCHES = {'edsr': '', 'spkres': '//'}


def load_csv(dir_path, model):
    """Load unit_counts.csv, filtering to the given model's rows."""
    csv_path = os.path.join(dir_path, 'unit_counts.csv')
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df[df['dataset'] == model].copy()


def find_dir(factor_dir, candidates, model):
    """Search for a config directory under sorter/{model}/factor_dir/."""
    for name in candidates:
        full = os.path.join(SORTER_DIR, model, factor_dir, name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, 'unit_counts.csv')):
            return full
    return None


def compute_metrics(df):
    """Return {sorter_name: {'recall': ..., 'precision': ..., 'f1': ...}}"""
    results = {}
    for _, row in df.iterrows():
        sorter = row['sorter_name']
        gt = float(row['num_gt'])
        detected = float(row['num_sorter'])
        well = float(row['num_well_detected'])
        fp = float(row['num_false_positive'])
        redundant = float(row['num_redundant'])
        overmerged = float(row['num_overmerged'])

        recall = well / gt if gt > 0 else 0 
        precision = well / detected if detected > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        results[sorter] = {'recall': recall, 'precision': precision, 'f1': f1}
    return results


def collect_all(baseline_dir, methods_dict):
    """Collect data for all models.
    Returns dict: {model: {factor: {'baseline': {sorter: metrics}, 'direct_transfer': ..., ...}}}
    """
    all_data = {}
    for model in MODELS:
        model_data = {}
        for factor_dir in FACTORS:
            factor_data = {}

            # baseline
            bpath = os.path.join(SORTER_DIR, model, factor_dir, baseline_dir)
            if os.path.isdir(bpath):
                df = load_csv(bpath, model)
                if df is not None:
                    factor_data['baseline'] = compute_metrics(df)

            # methods
            for method_name, candidates in methods_dict.items():
                mpath = find_dir(factor_dir, candidates, model)
                if mpath:
                    df = load_csv(mpath, model)
                    if df is not None:
                        factor_data[method_name] = compute_metrics(df)

            model_data[factor_dir] = factor_data
        all_data[model] = model_data
    return all_data


def get_avg_values(model_data, method_order, metric):
    """Return {method: [avg_factor16, avg_factor8, ...]} for one model."""
    all_methods = ['baseline'] + method_order
    result = {}
    for m in all_methods:
        result[m] = []
        for factor_dir in FACTORS:
            data = model_data.get(factor_dir, {})
            mdata = data.get(m, {})
            if mdata:
                vals = [v[metric] for v in mdata.values()]
                result[m].append(np.mean(vals))
            else:
                result[m].append(np.nan)
    return result


def build_avg_table(model_data, method_order, metric, model_label):
    """Build avg ± std table for one model."""
    rows = []
    for factor_dir in FACTORS:
        factor_label = int(factor_dir.replace('factor_', ''))
        data = model_data.get(factor_dir, {})
        row = {'Factor': factor_label}
        for m in ['baseline'] + method_order:
            mdata = data.get(m, {})
            if mdata:
                vals = [v[metric] for v in mdata.values()]
                row[m] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            else:
                row[m] = '---'
        rows.append(row)
    return pd.DataFrame(rows)


def build_detail_table(model_data, method_order, metric):
    """Per-sorter detail table for one model."""
    all_methods = ['baseline'] + method_order
    rows = []
    for factor_dir in FACTORS:
        factor_label = int(factor_dir.replace('factor_', ''))
        data = model_data.get(factor_dir, {})

        sorters = set()
        for mdata in data.values():
            sorters.update(mdata.keys())
        sorters = sorted(sorters)

        for sorter in sorters:
            row = {'Factor': factor_label, 'Sorter': sorter}
            for m in all_methods:
                mdata = data.get(m, {})
                row[m] = round(mdata[sorter][metric], 3) if sorter in mdata else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_model_figure(group_label, model, model_label, all_data, method_order):
    """Generate bar chart for a single model with all methods."""
    model_data = all_data.get(model, {})
    all_methods = ['baseline'] + method_order
    x = np.arange(len(FACTOR_DISPLAY))
    n_methods = len(all_methods)
    width = 0.12
    colors = ['#B0B0B0', '#C98F8F', "#7E9FC4", '#D4A87C', '#8CB896', '#C5A0C5']

    for metric in METRICS:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        avg_vals = get_avg_values(model_data, method_order, metric)
        for mi, method in enumerate(all_methods):
            vals = avg_vals.get(method, [np.nan] * len(FACTORS))
            offset = width * (mi - (n_methods - 1) / 2)
            ax.bar(x + offset, vals, width,
                   label=METHOD_LABELS.get(method, method),
                   color=colors[mi],
                   edgecolor='black', linewidth=0.3)

        ax.set_title(f'{model_label} — {metric.capitalize()}', fontsize=8)
        ax.set_xlabel('Bad Channel Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(FACTOR_DISPLAY)
        ax.set_ylim(0.2, 1.0)
        ax.legend(fontsize=8, loc='upper left', ncol=2, frameon=False, labelspacing=0.12,
                  bbox_to_anchor=(0.02, 1.02))

        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, f'cmp_methods_{model}_{group_label}_{metric}.tiff')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved figure: {out_path}")


def print_table(title, df):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(df.to_string(index=False))


def main():
    group_titles = {
        'np128': 'NP128 — Baseline vs Direct Transfer & Mitigation Methods',
        'sqmea': 'SQMEA — Baseline vs Direct Transfer & Mitigation Methods',
    }

    for group_label, baseline_dir, methods_dict in GROUPS:
        method_order = list(methods_dict.keys())
        group_title = group_titles[group_label]

        print(f"\n{'#'*90}")
        print(f"#  Group: {group_label}")
        print(f"#  Baseline: {baseline_dir}")
        print(f"{'#'*90}")

        all_data = collect_all(baseline_dir, methods_dict)

        for model in MODELS:
            model_data = all_data.get(model, {})
            model_label = MODEL_LABEL[model]
            print(f"\n  --- {model_label} ---")

            for metric in METRICS:
                df_detail = build_detail_table(model_data, method_order, metric)
                print_table(f"{group_label}/{model_label} — {metric} (per sorter)", df_detail)

                df_avg = build_avg_table(model_data, method_order, metric, model_label)
                print_table(f"{group_label}/{model_label} — {metric} (avg ± std)", df_avg)

                csv_path = os.path.join(TABLE_DIR, f'{model}_{group_label}_{metric}.csv')
                df_avg.to_csv(csv_path, index=False)
                print(f"  Saved (avg): {csv_path}")

                detail_csv_path = os.path.join(TABLE_DIR, f'{model}_{group_label}_{metric}_per_sorter.csv')
                df_detail.to_csv(detail_csv_path, index=False)
                print(f"  Saved (per sorter): {detail_csv_path}")

        for model in MODELS:
            plot_model_figure(group_label, model, MODEL_LABEL[model], all_data, method_order)

    print(f"\n{'='*90}")
    print("Done.")
    print(f"  Tables:  {TABLE_DIR}")
    print(f"  Figures: {FIG_DIR}")


if __name__ == '__main__':
    main()