# bartz/scripts/opt-plot.py
#
# Copyright (c) 2026, The Bartz Contributors
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Plot results produced by `opt.py`.

Reads ``config.jsonc`` and ``results.parquet`` from the directory produced by
``opt.py``, then writes PNGs back into the same directory.
"""

import colorsys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json5
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


@dataclass(frozen=True)
class Data:
    """Container for prepared data from `load_and_prepare_data`."""

    df: pl.DataFrame
    optimal_df: pl.DataFrame
    scan_col: str
    reduce_col: str
    matrix_cols: tuple[str, ...]
    fixed: dict[str, Any]
    out_dir: Path


def sanitize_for_filename(name: str) -> str:
    """Replace characters that are problematic in filenames."""
    return name.replace('=', '-').replace(', ', '_').replace(' ', '_').replace('/', '_')


def pick_scale(values: np.ndarray) -> str:
    """Pick 'log' if all finite values are positive and span more than a decade."""
    finite = values[np.isfinite(values)]
    if finite.size and (finite > 0).all() and finite.max() / finite.min() > 10:
        return 'log'
    return 'linear'


def save_fig(fig: plt.Figure, out_dir: Path) -> None:
    """Save figure to file with a status message."""
    fig_name = sanitize_for_filename(fig.get_label())
    save_file = out_dir / f'{fig_name}.png'
    print(f'write {save_file}...')
    fig.savefig(save_file, dpi=150)


def load_and_prepare_data(input_dir: Path) -> Data:
    """Load config and parquet, return df, optimal df, role info, fixed values."""
    config_path = input_dir / 'config.jsonc'
    results_path = input_dir / 'results.parquet'
    with config_path.open() as f:
        config = json5.load(f)

    scan_col = config['scan']
    reduce_col = config['reduce']
    matrix_cols = tuple(config['matrix'])

    df = pl.read_parquet(results_path)

    time_cols = {'time_est', 'time_lo', 'time_up'}
    required_base_cols = {scan_col, reduce_col, *time_cols}
    missing = required_base_cols - set(df.columns)
    if missing:
        msg = f'Missing required columns in {results_path}: {sorted(missing)}'
        raise ValueError(msg)

    # Identify fixed columns (everything left after role + time columns)
    role_cols = {scan_col, reduce_col, *matrix_cols}
    fixed_cols = [c for c in df.columns if c not in role_cols and c not in time_cols]
    fixed: dict[str, Any] = {}
    for c in fixed_cols:
        uniq = df[c].unique().to_list()
        if len(uniq) != 1:
            msg = f'Expected fixed column {c!r} to have one value, got {uniq}'
            raise ValueError(msg)
        fixed[c] = uniq[0]

    # Treat null values of the reduce param as 0.5 (so they're visible on log scale)
    df = df.with_columns(pl.col(reduce_col).fill_null(0.5))

    # Build a per-row label from the matrix params
    if matrix_cols:
        label_exprs = [
            pl.concat_str([pl.lit(f'{c}='), pl.col(c).cast(pl.Utf8).fill_null('null')])
            for c in matrix_cols
        ]
        df = df.with_columns(pl.concat_str(label_exprs, separator=', ').alias('label'))
    else:
        df = df.with_columns(pl.lit('default').alias('label'))

    # For each combination of matrix params and scan param, pick the reduce
    # value that minimises time_est
    group_cols = [*matrix_cols, scan_col, 'label']
    optimal_df = df.group_by(group_cols, maintain_order=True).agg(
        pl.col(reduce_col).sort_by('time_est').first().alias(f'opt_{reduce_col}')
    )

    return Data(
        df=df,
        optimal_df=optimal_df,
        scan_col=scan_col,
        reduce_col=reduce_col,
        matrix_cols=matrix_cols,
        fixed=fixed,
        out_dir=input_dir,
    )


def add_fixed_textbox(ax: plt.Axes, fixed: dict[str, Any]) -> None:
    """Annotate axes with a small text box listing the fixed param values."""
    if not fixed:
        return
    text = 'fixed:\n' + '\n'.join(f'  {k}={v}' for k, v in fixed.items())
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=8,
        family='monospace',
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )


def color_for_matrix_indices(indices: list[int], dim_sizes: list[int]) -> tuple:
    """Pick an RGB color from a multi-dim categorical index.

    Hue from the first matrix dim; saturation/brightness from the second; any
    extra dims are not visualised in colour (the legend distinguishes them).
    """
    if not indices:
        return (0.2, 0.4, 0.8)
    hue = indices[0] / max(1, dim_sizes[0])
    if len(indices) >= 2:
        sat = 0.05 + 0.90 * (indices[1] / max(1, dim_sizes[1] - 1 or 1))
        bri = 0.95 - 0.45 * (indices[1] / max(1, dim_sizes[1] - 1 or 1))
    else:
        sat = 0.85
        bri = 0.65
    return colorsys.hsv_to_rgb(hue, sat, bri)


def plot_optimal(data: Data, fig_name_prefix: str) -> None:
    """Plot optimal `reduce` value vs `scan` value."""
    fig, ax = plt.subplots(
        num=f'{fig_name_prefix}_optimal',
        figsize=(10, 6),
        layout='constrained',
        clear=True,
    )

    # Map each matrix dim's unique values to a position index
    hp_to_idx: list[dict[Any, int]] = []
    dim_sizes: list[int] = []
    for hp_name in data.matrix_cols:
        values = data.optimal_df[hp_name].unique().sort()
        hp_to_idx.append({v: i for i, v in enumerate(values)})
        dim_sizes.append(len(values))

    group_cols = [*data.matrix_cols, 'label']
    for group_keys, subset in data.optimal_df.group_by(group_cols, maintain_order=True):
        hp_vals = group_keys[:-1]
        label = group_keys[-1]
        indices = [hp_to_idx[i][hp_vals[i]] for i in range(len(data.matrix_cols))]
        color = color_for_matrix_indices(indices, dim_sizes)

        scan_values = subset[data.scan_col].to_numpy()
        opt_values = subset[f'opt_{data.reduce_col}'].to_numpy()
        ax.plot(
            scan_values, opt_values, marker='o', label=label, markersize=4, color=color
        )

    scan_all = data.optimal_df[data.scan_col].to_numpy().astype(float)
    opt_all = data.optimal_df[f'opt_{data.reduce_col}'].to_numpy().astype(float)
    ax.set(
        xscale=pick_scale(scan_all),
        yscale=pick_scale(opt_all),
        xlabel=data.scan_col,
        ylabel=f'optimal {data.reduce_col}',
        title=f'Optimal {data.reduce_col} vs {data.scan_col}',
    )
    if data.matrix_cols:
        ax.legend()
    ax.grid(True, alpha=0.3)
    add_fixed_textbox(ax, data.fixed)

    save_fig(fig, data.out_dir)


def plot_time_vs_reduce_series(data: Data, fig_name_prefix: str) -> None:
    """Plot time vs reduce param for each scan value, one figure per matrix combo."""
    scan_values = (
        data.df.select(data.scan_col).unique().sort(data.scan_col).to_series().to_list()
    )

    for (label,), subset in data.df.group_by('label', maintain_order=True):
        fig, ax = plt.subplots(
            num=f'{fig_name_prefix}_{label}',
            figsize=(10, 6),
            layout='constrained',
            clear=True,
        )

        brightness = np.linspace(0.95, 0.05, len(scan_values))

        for scan_val, gray in zip(scan_values, brightness, strict=True):
            n_subset = subset.filter(pl.col(data.scan_col) == scan_val).sort(
                data.reduce_col
            )
            if n_subset.height == 0:
                continue
            reduce_vals = n_subset[data.reduce_col].to_numpy()
            time_est = n_subset['time_est'].to_numpy()
            min_time = time_est.min()
            time_lo = n_subset['time_lo'].to_numpy()
            time_up = n_subset['time_up'].to_numpy()
            ax.fill_between(
                reduce_vals,
                time_lo / min_time,
                time_up / min_time,
                color=str(gray),
                label=f'{data.scan_col}={scan_val}',
            )
            min_idx = time_est.argmin()
            ax.plot(
                reduce_vals[min_idx],
                time_est[min_idx] / min_time,
                marker='o',
                mfc='none',
                mec=[1, gray, gray],
                linestyle='none',
            )

        reduce_all = subset[data.reduce_col].to_numpy().astype(float)
        ax.set(
            xscale=pick_scale(reduce_all),
            xlabel=data.reduce_col,
            ylabel='time range / min(time_est)',
            title=f'Time vs {data.reduce_col}',
            ylim=(0, None),
        )
        ax.legend(title=label.replace(', ', '\n'), loc='upper right')
        ax.grid(True, alpha=0.3)
        add_fixed_textbox(ax, data.fixed)

        save_fig(fig, data.out_dir)


def parse_args() -> Namespace:
    """Define and parse command line arguments."""
    parser = ArgumentParser(
        description='Plot optimal reduce-param vs scan-param and time curves.'
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Directory produced by opt.py (containing config.jsonc and results.parquet).',
    )
    return parser.parse_args()


def main() -> None:
    """Entry point of the script."""
    args = parse_args()
    input_prefix = args.input_dir.name
    data = load_and_prepare_data(args.input_dir)

    plot_optimal(data, input_prefix)
    plot_time_vs_reduce_series(data, input_prefix)

    plt.show()


if __name__ == '__main__':
    main()
