"""Generate the Capriccio dataset.

This script takes as input the path to the Sentiment140 dataset
and slices it into 38 overlapping slices. Each slice has 425000
tweets for training (%d_train.json) and 75000 tweets for eval
(%d_val.json).

Usage:
    python generate.py /path/to/sentiment140.json
"""

import argparse
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import datasets
import numpy as np
import pandas as pd


def main(output_dir: str) -> None:
    """Run the main routine."""
    # Prepare raw dataset
    print("Preparing raw dataset.")
    df = datasets.load_dataset("sentiment140")["train"].to_pandas()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        df["date"] = pd.to_datetime(df.date)
    df["ind"] = df.index
    df = df.set_index("date")
    df = df.sort_index()
    df = df.rename({"sentiment": "label"}, axis=1)
    df = df.drop(["user", "query"], axis=1)

    # Slice datasets
    stride = timedelta(days=1)
    size = 500_000

    print(f"Slicing datasets with stride {stride} and size {size}.")

    slice_index = 1
    sliced = []
    # Skip April since there are too many days with no tweets.
    now = datetime(year=2009, month=5, day=1)
    end = df.index.max()
    while now < end:
        loc = df.index.get_loc(now.strftime("%m/%d/%Y")).start
        slice_df = df.iloc[loc : loc + size]
        if len(slice_df) < size:
            break

        # Compute sample overlap ratio
        if sliced:
            overlap = len(sliced[-1].merge(slice_df, how="inner", on="ind"))
            print(
                f"{slice_index:2d}: {slice_df.index.min()} ~ {slice_df.index.max()}, overlap = {overlap/size:.3f}"
            )
        else:
            print(f"{slice_index:2d}: {slice_df.index.min()} ~ {slice_df.index.max()}")

        sliced.append(slice_df)

        slice_index += 1
        now += stride

    print(f"{len(sliced)} datasets of size {size} were created.")

    # Split train and validation sets and save
    print("Sampling validation set and saving.")
    seed = 42
    train_frac = 0.85
    save_dir = Path(output_dir)
    os.makedirs(save_dir, exist_ok=True)

    for slice_index, dataset in enumerate(sliced):
        ind = np.random.default_rng(seed).permutation(len(dataset))
        shuffled = dataset.iloc[ind]
        boundary = int(len(dataset) * train_frac)
        train = shuffled.iloc[:boundary].drop(["ind"], axis=1).reset_index()
        val = shuffled.iloc[boundary:].drop(["ind"], axis=1).reset_index()
        train.to_json(
            save_dir / f"{slice_index+1}_train.json", orient="records", lines=True
        )
        val.to_json(
            save_dir / f"{slice_index+1}_val.json", orient="records", lines=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", "-o", required=True, help="Directory to save Capriccio"
    )
    args = parser.parse_args()

    main(args.output_dir)
