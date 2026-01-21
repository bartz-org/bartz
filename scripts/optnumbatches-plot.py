# bartz/scripts/optnumbatches-plot.py
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

"""Plot the results produced by optnumbatches.py."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def sanitize_for_filename(name: str) -> str:
    """Replace characters that are problematic in filenames."""
    return name.replace('=', '-').replace(', ', '_').replace(' ', '_').replace('/', '_')


def save_fig(fig: plt.Figure) -> None:
    """Save figure to file with a status message."""
    fig_save_dir = Path(__file__).parent
    fig_name = sanitize_for_filename(fig.get_label())
    save_file = fig_save_dir / f'{fig_name}.png'
    print(f'write {save_file}...')
    fig.savefig(save_file, dpi=150)


def load_and_prepare_data(
    input_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str], str]:
    """Load parquet data and return full/optimal datasets plus labels."""
    df = pl.read_parquet(input_path)

    required_base_cols = {'n', 'time_est', 'time_lo', 'time_up'}
    missing_base_cols = required_base_cols - set(df.columns)
    if missing_base_cols:
        missing_list = ', '.join(sorted(missing_base_cols))
        msg = f'Missing required columns: {missing_list}'
        raise ValueError(msg)

    if 'num_batches' in df.columns:
        batch_col_name = 'num_batches'
    elif 'num_trees_times_num_batches' in df.columns:
        batch_col_name = 'num_trees_times_num_batches'
        df = df.rename({'num_trees_times_num_batches': 'num_batches'})
    else:
        msg = 'Missing required column: num_batches or num_trees_times_num_batches'
        raise ValueError(msg)

    # Treat null num_batches as 0.5 (so it's visible on log scale)
    df = df.with_columns(pl.col('num_batches').fill_null(0.5))

    # Drop rows where num_batches exceeds n
    df = df.filter(pl.col('num_batches') <= pl.col('n'))

    required_cols = {*required_base_cols, 'num_batches'}
    hyperparam_cols = [c for c in df.columns if c not in required_cols]

    # Create a label for each combination of hyperparameters
    if hyperparam_cols:
        label_exprs = [
            pl.concat_str([pl.lit(f'{c}='), pl.col(c).cast(pl.Utf8).fill_null('null')])
            for c in hyperparam_cols
        ]
        df = df.with_columns(pl.concat_str(label_exprs, separator=', ').alias('label'))
    else:
        df = df.with_columns(pl.lit('default').alias('label'))

    # For each combination of hyperparameters and n, find the num_batches that minimizes time_est
    group_cols = [*hyperparam_cols, 'n', 'label']
    optimal_df = (
        df.group_by(group_cols)
        .agg(pl.col('num_batches').sort_by('time_est').first().alias('opt_num_batches'))
        .sort(group_cols)
    )

    labels = df.select('label').unique().sort('label').to_series().to_list()

    return df, optimal_df, labels, batch_col_name


def plot_optimal_num_batches(
    optimal_df: pl.DataFrame,
    labels: list[str],
    batch_col_name: str,
    fig_name_prefix: str,
) -> None:
    """Plot optimal num_batches vs n."""
    fig, ax = plt.subplots(
        num=fig_name_prefix, figsize=(10, 6), layout='constrained', clear=True
    )

    for label in labels:
        subset = optimal_df.filter(pl.col('label') == label)
        n_values = subset['n'].to_numpy()
        opt_num_batches = subset['opt_num_batches'].to_numpy()
        ax.plot(n_values, opt_num_batches, marker='o', label=label, markersize=4)

    ax.set(
        xscale='log',
        yscale='log',
        xlabel='n',
        ylabel=f'optimal {batch_col_name}',
        title=f'Optimal {batch_col_name} vs n',
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig(fig)


def plot_time_vs_num_batches_series(
    df: pl.DataFrame, labels: list[str], batch_col_name: str, fig_name_prefix: str
) -> None:
    """Plot time vs num_batches for each n, one figure per hyperparameter combo."""
    n_values = df.select('n').unique().sort('n').to_series().to_list()

    for label in labels:
        subset = df.filter(pl.col('label') == label)

        fig, ax = plt.subplots(
            num=f'{fig_name_prefix}_{label}',
            figsize=(10, 6),
            layout='constrained',
            clear=True,
        )

        brightness = np.linspace(0.95, 0.05, len(n_values))

        for n, gray in zip(n_values, brightness, strict=True):
            n_subset = subset.filter(pl.col('n') == n).sort('num_batches')
            if n_subset.height == 0:
                continue
            num_batches = n_subset['num_batches'].to_numpy()
            time_est = n_subset['time_est'].to_numpy()
            min_time = time_est.min()
            time_lo = n_subset['time_lo'].to_numpy()
            time_up = n_subset['time_up'].to_numpy()
            ax.fill_between(
                num_batches,
                time_lo / min_time,
                time_up / min_time,
                color=str(gray),
                label=f'n={n}',
            )
            min_idx = time_est.argmin()
            ax.plot(
                num_batches[min_idx],
                time_est[min_idx] / min_time,
                marker='o',
                mfc='none',
                mec=[1, gray, gray],
                linestyle='none',
            )

        ax.set(
            xscale='log',
            xlabel=f'{batch_col_name}',
            ylabel='time range / min(time_est)',
            title='Time vs num batches',
            ylim=(0, None),
        )
        ax.legend(title=label.replace(', ', '\n'), loc='upper right')
        ax.grid(True, alpha=0.3)

        save_fig(fig)


def parse_args() -> Namespace:
    """Define and parse command line arguments."""
    parser = ArgumentParser(
        description='Plot optimal num_batches vs n and time curves.'
    )
    parser.add_argument(
        'input_path', type=Path, help='Path to the optnumbatches parquet file.'
    )
    return parser.parse_args()


def main() -> None:
    """Entry point of the script."""
    args = parse_args()
    input_prefix = args.input_path.stem
    df, optimal_df, labels, batch_col_name = load_and_prepare_data(args.input_path)

    plot_optimal_num_batches(optimal_df, labels, batch_col_name, input_prefix)
    plot_time_vs_num_batches_series(df, labels, batch_col_name, input_prefix)

    plt.show()


if __name__ == '__main__':
    main()
