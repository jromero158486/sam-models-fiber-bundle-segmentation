"""Inspect the groups and arrays in an OME-Zarr store."""

from __future__ import annotations

import argparse
from pathlib import Path

import zarr


def walk(group, indent: int = 0) -> None:
    """Print an OME-Zarr hierarchy without loading array contents."""
    prefix = "  " * indent
    print(f"{prefix}[GROUP] {group.path or '/'}")
    for key in sorted(group.array_keys()):
        array = group[key]
        print(f"{prefix}  [ARRAY] {key}: shape={array.shape}, dtype={array.dtype}")
    for key in sorted(group.group_keys()):
        walk(group[key], indent + 1)


def parse_args() -> argparse.Namespace:
    """Parse the OME-Zarr path supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to an .ome.zarr store")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.path.exists():
        raise FileNotFoundError(f"OME-Zarr store does not exist: {args.path}")
    walk(zarr.open_group(args.path, mode="r"))


if __name__ == "__main__":
    main()

