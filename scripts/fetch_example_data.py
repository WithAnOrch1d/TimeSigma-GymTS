"""Obtain pinned TODS demo data or all seven full-experiment datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


SOURCE = "https://github.com/CSTCloudOps/datasets.git"
REVISION = "0c6dfa195a5af8ef75938bc1ca6b870e49c7b14a"
ROOT = Path(__file__).resolve().parents[1]
FULL_DATASETS = ('TODS', 'NAB', 'NEK', 'UCR', 'WSD', 'Yahoo', 'AIOPS')


def git(destination: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(destination), *args], check=True,
        capture_output=True, text=True, timeout=300,
    ).stdout.strip()


def verify(destination: Path, datasets=('TODS',)) -> None:
    if (git(destination, "rev-parse", "HEAD") != REVISION
            or git(destination, "config", "--get", "remote.origin.url") != SOURCE
            or git(destination, "status", "--short")):
        raise ValueError("data source must be the pinned, clean upstream checkout")
    for dataset in datasets:
        directory = destination / 'UTS' / dataset
        if not directory.is_dir() or not any(directory.iterdir()):
            raise FileNotFoundError(f'missing dataset: {dataset}')
        for curve in directory.iterdir():
            if not curve.is_dir():
                continue
            for name in ('train.npy', 'test.npy', 'train_label.npy', 'test_label.npy', 'info.json'):
                if not (curve / name).is_file():
                    raise FileNotFoundError(f'missing {dataset}/{curve.name}/{name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "datasets/easytsad")
    parser.add_argument('--full', action='store_true', help='Fetch six development datasets plus AIOPS')
    args = parser.parse_args()
    destination = args.destination.resolve()
    datasets = FULL_DATASETS if args.full else ('TODS',)
    if destination.exists():
        verify(destination)  # Refuse dirty/unpinned checkout before expanding it.
        if args.full and not all((destination / 'UTS' / d).is_dir() for d in datasets):
            git(destination, 'sparse-checkout', 'set', *(f'UTS/{d}' for d in datasets))
        verify(destination, datasets)
        print(f"Existing pinned data verified: {destination}")
        return
    destination.mkdir(parents=True)
    # Incomplete downloads are retained for inspection; no reset or deletion.
    git(destination, "init")
    git(destination, "remote", "add", "origin", SOURCE)
    git(destination, "sparse-checkout", "init", "--cone")
    git(destination, "sparse-checkout", "set", *(f'UTS/{d}' for d in datasets))
    git(destination, "fetch", "--depth=1", "--filter=blob:none", "origin", REVISION)
    git(destination, "checkout", "--detach", "FETCH_HEAD")
    verify(destination, datasets)
    print(f"Pinned data ready: {destination}")


if __name__ == "__main__":
    main()
