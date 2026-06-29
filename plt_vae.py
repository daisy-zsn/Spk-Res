#!/usr/bin/env python3
"""
Compare VAE vs SpkRes reconstruction on spike sorting performance.

Output: tables and figures saved under tables/ and figures/.
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

# Groups: (label, title, vae_config, spkres_config)
GROUPS = [
    ('np', 'NP — VAE vs SpkRes',
     {'model': 'edsr', 'config': 'single_np_rd_nm_p1_5_vae'},
     {'model': 'spkres', 'config': 'single_np_rd_nm'}),
    ('sq', 'SQ — VAE vs SpkRes',
     {'model': 'edsr', 'config': 'single_sq_rd_nm_p1_5_vae'},
     {'model': 'spkres', 'config': 'single_sq_rd_nm'}),
]

COLORS = {'VAE': '#D4A87C', 'SpkRes': '#BC8F8F'}
MARKERS = {'VAE': 's', 'SpkRes': 'D'}
ORDER = ['VAE', 'SpkRes']


def load_csv(dir_path, model):
    """Load unit_counts.csv, filtering to the given model's rows."""
    csv_path = os.path.join(dir_path, 'unit_counts.csv')
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df[df['dataset'] == model].copy()


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


def collect_data(specs):
    """Collect data for multiple methods.
    specs = {'VAE': {'model': 'edsr', 'config': '...'}, 'SpkRes': {...}}
    Returns dict: {factor: {'VAE': {sorter: metrics}, 'SpkRes': {...}}}
    """
    all_data = {}
    for factor_dir in FACTORS:
        factor_data = {}
        for label, spec in specs.items():
            path = os.path.join(SORTER_DIR, spec['model'], factor_dir, spec['config'])
            if os.path.isdir(path):
                df = load_csv(path, spec['model'])
                if df is not None:
                    factor_data[label] = compute_metrics(df)
        all_data[factor_dir] = factor_data
    return all_data


def build_avg_table(all_data, metric):
    """rows = Factor, cols = VAE / SpkRes, cells = avg ± std across sorters."""
    rows = []
    for factor_dir in FACTORS:
        factor_label = int(factor_dir.replace('factor_', ''))
        data = all_data.get(factor_dir, {})
        row = {'Factor': factor_label}
        for method in ORDER:
            mdata = data.get(method, {})
            if mdata:
                vals = [v[metric] for v in mdata.values()]
                row[method] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            else:
                row[method] = '---'
        rows.append(row)
    return pd.DataFrame(rows)


def build_detail_table(all_data, metric):
    """rows = Factor x Sorter, cols = VAE / SpkRes"""
    rows = []
    for factor_dir in FACTORS:
        factor_label = int(factor_dir.replace('factor_', ''))
        data = all_data.get(factor_dir, {})

        sorters = set()
        for mdata in data.values():
            sorters.update(mdata.keys())
        sorters = sorted(sorters)

        for sorter in sorters:
            row = {'Factor': factor_label, 'Sorter': sorter}
            for method in ORDER:
                mdata = data.get(method, {})
                row[method] = round(mdata[sorter][metric], 3) if sorter in mdata else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def get_avg_values(all_data, metric):
    """Return {method: [avg_factor16, avg_factor8, avg_factor4, avg_factor2]}"""
    result = {m: [] for m in ORDER}
    for factor_dir in FACTORS:
        data = all_data.get(factor_dir, {})
        for method in ORDER:
            mdata = data.get(method, {})
            if mdata:
                vals = [v[metric] for v in mdata.values()]
                result[method].append(np.mean(vals))
            else:
                result[method].append(np.nan)
    return result


def plot_all_groups_2x3(all_data_np, all_data_sq):
    """3 rows × 2 columns line chart: rows=recall/precision/f1, cols=NP/SQ."""
    x = np.arange(len(FACTOR_DISPLAY))
    datasets = [
        ('Neuropixels-128', all_data_np),
        ('SqMEA-10-15', all_data_sq),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(6, 6))

    for row_idx, metric in enumerate(METRICS):
        for col_idx, (ds_label, all_data) in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            avg_vals = get_avg_values(all_data, metric)

            for method in ORDER:
                vals = avg_vals[method]
                ax.plot(x, vals, marker=MARKERS[method], linestyle='-',
                        color=COLORS[method], linewidth=1, markersize=4)

            ax.set_title(f'{ds_label} — {metric.capitalize()}', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(FACTOR_DISPLAY)

            if row_idx == len(METRICS) - 1:
                ax.set_xlabel('Bad Channel Ratio')
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if metric == 'precision':
                if col_idx == 0:
                    ax.set_ylim(0.2, 1.0)
                else:
                    ax.set_ylim(0.2, 1.0)
            elif metric == 'recall':
                if col_idx == 1:
                    ax.set_ylim(0.0, 0.8)
                else:
                    ax.set_ylim(0.3, 1.0)
            else:
                ax.set_ylim(0.2, 1.0)
        

    handles = [plt.Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], linestyle='-',
                          linewidth=1, markersize=4) for m in ORDER]
    fig.legend(handles, ORDER, fontsize=8, loc='upper right', bbox_to_anchor=(0.98, 0.96),
               ncol=1, frameon=False, labelspacing=0.18)

    plt.tight_layout(pad=0.8)
    out_path = os.path.join(FIG_DIR, 'cmp_vae_spkres.tiff')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {out_path}")


def print_table(title, df):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(df.to_string(index=False))


def main():
    all_data_dict = {}
    for group_label, group_title, vae_spec, spkres_spec in GROUPS:
        specs = {'VAE': vae_spec, 'SpkRes': spkres_spec}
        print(f"\n{'#'*80}")
        print(f"#  Group: {group_label} — {group_title}")
        print(f"#  VAE: {vae_spec['config']}  |  SpkRes: {spkres_spec['config']}")
        print(f"{'#'*80}")

        all_data = collect_data(specs)
        all_data_dict[group_label] = all_data

        for metric in METRICS:
            df_detail = build_detail_table(all_data, metric)
            print_table(f"{group_label} — {metric} (per sorter)", df_detail)

            df_avg = build_avg_table(all_data, metric)
            print_table(f"{group_label} — {metric} (avg ± std)", df_avg)

            csv_path = os.path.join(TABLE_DIR, f'vae_vs_spkres_{group_label}_{metric}.csv')
            df_avg.to_csv(csv_path, index=False)
            print(f"  Saved (avg): {csv_path}")

            detail_csv_path = os.path.join(TABLE_DIR, f'vae_vs_spkres_{group_label}_{metric}_per_sorter.csv')
            df_detail.to_csv(detail_csv_path, index=False)
            print(f"  Saved (per sorter): {detail_csv_path}")

    plot_all_groups_2x3(all_data_dict['np'], all_data_dict['sq'])

    print(f"\n{'='*80}")
    print("Done.")
    print(f"  Tables:  {TABLE_DIR}")
    print(f"  Figures: {FIG_DIR}")


if __name__ == '__main__':
    main()