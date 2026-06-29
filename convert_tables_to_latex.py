#!/usr/bin/env python3
"""
Convert all CSV tables in tables/ to LaTeX .tex files.
Output: tables_latex/ directory with .tex files.
"""
import os
import csv
import pandas as pd

TABLE_DIR = 'tables'
OUT_DIR = 'tables_latex'
os.makedirs(OUT_DIR, exist_ok=True)


def csv_to_latex(csv_path, out_path):
    """Convert a single CSV to a LaTeX tabular."""
    df = pd.read_csv(csv_path)
    n_cols = len(df.columns)
    col_str = 'l' + 'c' * (n_cols - 1)

    lines = []
    lines.append('\\begin{table}[htbp]')
    lines.append('  \\centering')
    lines.append(f'  \\caption{{Table from {os.path.basename(csv_path)}}}')
    lines.append(f'  \\label{{tab:{os.path.splitext(os.path.basename(csv_path))[0]}}}')
    lines.append(f'  \\begin{{tabular}}{{{col_str}}}')
    lines.append('    \\toprule')

    # Header
    header = ' & '.join(df.columns)
    lines.append(f'    {header} \\\\')
    lines.append('    \\midrule')

    # Rows
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            v = row[col]
            if pd.isna(v):
                vals.append('---')
            else:
                vals.append(str(v))
        lines.append(f'    {" & ".join(vals)} \\\\')

    lines.append('    \\bottomrule')
    lines.append('  \\end{tabular}')
    lines.append('\\end{table}')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved: {out_path}")


def main():
    csv_files = sorted([f for f in os.listdir(TABLE_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} CSV files.")

    for csv_file in csv_files:
        csv_path = os.path.join(TABLE_DIR, csv_file)
        out_name = os.path.splitext(csv_file)[0] + '.tex'
        out_path = os.path.join(OUT_DIR, out_name)
        csv_to_latex(csv_path, out_path)

    print(f"\nDone. {len(csv_files)} LaTeX tables saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()