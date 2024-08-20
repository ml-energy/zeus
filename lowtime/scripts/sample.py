"""Sample a subset of frequency assignments.

Given an input directory, goes throuh all .py files in order of file names and samples every n-th file.
The output directory is created and should not already exist. Sampled frequency files are copied into the output directory.
"""

import os
import shutil
import argparse


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input directory")
    parser.add_argument("output", help="output directory")
    parser.add_argument("-n", type=int, default=50, help="sample every n-th frequency assignment")
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Sample frequency files."""
    os.makedirs(args.output)
    files = sorted(list(filter(lambda x: x.endswith(".py"), os.listdir(args.input))))
    num_sampled = 0
    for i, filename in enumerate(files):
        # Make sure we include the last one.
        if i % args.n == 0 or i == len(files) - 1:
            shutil.copy(os.path.join(args.input, filename), args.output)
            num_sampled += 1

    print(
        f"Sampled {num_sampled} frequency assignments out of {len(files)} and saved them to {args.output}",
    )


if __name__ == "__main__":
    main(parse_args())
