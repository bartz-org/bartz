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

import colorsys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


@dataclass(frozen=True)
class Data:
    """Container for prepared data from load_and_prepare_data."""

    df: pl.DataFrame
    optimal_df: pl.DataFrame
    hyperparam_cols: tuple[str, ...]


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


def load_and_prepare_data(input_path: Path) -> Data:
    """Load parquet data and return full/optimal datasets with metadata."""
    df = pl.read_parquet(input_path)

    required_base_cols = {'n', 'time_est', 'time_lo', 'time_up', 'num_batches'}
    missing_base_cols = required_base_cols - set(df.columns)
    if missing_base_cols:
        missing_list = ', '.join(sorted(missing_base_cols))
        msg = f'Missing required columns: {missing_list}'
        raise ValueError(msg)

    # Treat null num_batches as 0.5 (so it's visible on log scale)
    df = df.with_columns(pl.col('num_batches').fill_null(0.5))

    # Drop rows where num_batches exceeds n
    df = df.filter(pl.col('num_batches') <= pl.col('n'))

    required_cols = required_base_cols
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
    optimal_df = df.group_by(group_cols, maintain_order=True).agg(
        pl.col('num_batches').sort_by('time_est').first().alias('opt_num_batches')
    )

    return Data(df=df, optimal_df=optimal_df, hyperparam_cols=tuple(hyperparam_cols))


def plot_optimal_num_batches(data: Data, fig_name_prefix: str) -> None:
    """Plot optimal num_batches vs n."""
    assert len(data.hyperparam_cols) == 2, (
        f'Expected exactly 2 hyperparams, got {len(data.hyperparam_cols)}'
    )

    fig, ax = plt.subplots(
        num=f'{fig_name_prefix}_optimal',
        figsize=(10, 6),
        layout='constrained',
        clear=True,
    )

    # Get unique sorted values for each hyperparam and create index mappings
    hp_to_idx = []
    for hp_name in data.hyperparam_cols:
        values = data.optimal_df[hp_name].unique().sort()
        hp_to_idx.append({v: i for i, v in enumerate(values)})

    # Create hue and brightness arrays
    hues = np.linspace(0, 1, len(hp_to_idx[0]), endpoint=False)
    saturation = np.linspace(0.05, 0.95, len(hp_to_idx[1]))
    brightness = np.linspace(0.95, 0.50, len(hp_to_idx[1]))

    group_cols = [*data.hyperparam_cols, 'label']
    for group_keys, subset in data.optimal_df.group_by(group_cols, maintain_order=True):
        # Extract hyperparam values and label from group_keys
        hp_vals = group_keys[:-1]
        label = group_keys[-1]

        # Get indices for each hyperparam
        hp_indices = [
            hp_to_idx[i][hp_vals[i]] for i in range(len(data.hyperparam_cols))
        ]

        hue = hues[hp_indices[0]]
        sat = saturation[hp_indices[1]]
        bri = brightness[hp_indices[1]]
        color = colorsys.hsv_to_rgb(hue, sat, bri)

        n_values = subset['n'].to_numpy()
        opt_num_batches = subset['opt_num_batches'].to_numpy()
        ax.plot(
            n_values,
            opt_num_batches,
            marker='o',
            label=label,
            markersize=4,
            color=color,
        )

    ax.set(
        xscale='log',
        yscale='log',
        xlabel='n',
        ylabel='optimal num_batches',
        title='Optimal num_batches vs n',
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig(fig)


def plot_time_vs_num_batches_series(data: Data, fig_name_prefix: str) -> None:
    """Plot time vs num_batches for each n, one figure per hyperparameter combo."""
    n_values = data.df.select('n').unique().sort('n').to_series().to_list()

    for (label,), subset in data.df.group_by('label', maintain_order=True):
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
            xlabel='num_batches',
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
    data = load_and_prepare_data(args.input_path)

    plot_optimal_num_batches(data, input_prefix)
    plot_time_vs_num_batches_series(data, input_prefix)

    plt.show()


if __name__ == '__main__':
    main()
