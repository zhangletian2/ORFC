"""Build the full-5k blk05 cache as train-4500 followed by val-500."""

import argparse
from pathlib import Path

import numpy as np

from .config import activate


def merge(target, sources):
    arrays = [np.load(path, mmap_mode="r") for path in sources]
    shape = (sum(len(array) for array in arrays), *arrays[0].shape[1:])
    if target.exists():
        current = np.load(target, mmap_mode="r")
        if current.shape != shape or current.dtype != arrays[0].dtype:
            raise SystemExit(f"existing cache has wrong contract: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=arrays[0].dtype, shape=shape)
    first = 0
    for array in arrays:
        for start in range(0, len(array), 16):
            stop = min(start + 16, len(array))
            output[first + start:first + stop] = array[start:stop]
        first += len(array)
    output.flush()
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="")
    args = parser.parse_args()
    C = activate("blk05")
    root = Path(args.cache) if args.cache else C.CACHE
    merge(C.TRAIN_FEATURES, [
        root / "features_train_blk05_n4500_ss20260730.npy",
        root / "features_val_blk05_n500_ss20260730.npy"])
    merge(C.TRAIN_TEACHERS, [
        root / "teacher_train_blk05_n4500_ss20260730.npy",
        root / "teacher_val_blk05_n500_ss20260730.npy"])
    print(C.TRAIN_FEATURES)
    print(C.TRAIN_TEACHERS)


if __name__ == "__main__":
    main()
