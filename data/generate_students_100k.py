#!/usr/bin/env python3
"""
Generate the 100k benchmark dataset used by the index-structure experiments.

Input:
    Kaggle National Names CSV, commonly named NationalNames.csv.

Expected useful columns:
    Name / name
    Count / count

Output:
    CSV with schema:
        name,reg_no,cgpa

The generated CGPA value is synthetic. It is created by log-scaling the
source Count field into the range [0, 10]. It is intended only to provide
a controlled numeric lookup attribute for FQA benchmarking; it is not real
academic CGPA data.

Example:
    python data/generate_students_100k.py \
        --input NationalNames.csv \
        --output data/students_100k.csv \
        --n 100000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def normalize_header(header: str) -> str:
    """Normalize a CSV header for case-insensitive matching."""
    return header.strip().lower().replace(" ", "_")


def find_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    """Find a source CSV column by trying normalized candidate names."""
    normalized_to_original = {normalize_header(name): name for name in fieldnames}
    for candidate in candidates:
        key = normalize_header(candidate)
        if key in normalized_to_original:
            return normalized_to_original[key]
    raise ValueError(
        f"Could not find any of these columns: {', '.join(candidates)}. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def read_name_count_rows(input_path: Path) -> List[Tuple[str, int]]:
    """Read (name, count) rows from the National Names CSV."""
    rows: List[Tuple[str, int]] = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        name_col = find_column(reader.fieldnames, ["Name", "name"])
        count_col = find_column(reader.fieldnames, ["Count", "count"])

        for row in reader:
            name = (row.get(name_col) or "").strip()
            count_raw = (row.get(count_col) or "").strip()

            if not name or not count_raw:
                continue

            try:
                count = int(float(count_raw))
            except ValueError:
                continue

            if count <= 0:
                continue

            rows.append((name, count))

    if not rows:
        raise ValueError("No valid (name, count) rows found in input CSV.")

    return rows


def log_scaled_values(counts: List[int]) -> List[float]:
    """Scale counts into [0, 10] using log1p normalization."""
    logs = [math.log1p(c) for c in counts]
    lo = min(logs)
    hi = max(logs)

    if math.isclose(lo, hi):
        return [10.0 for _ in logs]

    scaled = []
    for value in logs:
        cgpa = 10.0 * (value - lo) / (hi - lo)
        cgpa = max(0.0, min(10.0, cgpa))
        scaled.append(cgpa)
    return scaled


def generate_records(rows: List[Tuple[str, int]], n: int, seed: int) -> List[Dict[str, str]]:
    """Shuffle rows deterministically and create benchmark records."""
    if n <= 0:
        raise ValueError("--n must be positive.")

    rng = random.Random(seed)
    rows_copy = list(rows)
    rng.shuffle(rows_copy)

    if len(rows_copy) < n:
        raise ValueError(
            f"Requested {n} records, but input contains only {len(rows_copy)} valid rows."
        )

    selected = rows_copy[:n]
    counts = [count for _, count in selected]
    cgpas = log_scaled_values(counts)

    records: List[Dict[str, str]] = []
    for idx, ((name, _count), cgpa) in enumerate(zip(selected, cgpas), start=1):
        records.append(
            {
                "name": name,
                "reg_no": f"REG{idx:06d}",
                "cgpa": f"{cgpa:.6f}",
            }
        )

    return records


def write_output(records: List[Dict[str, str]], output_path: Path) -> None:
    """Write generated records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "reg_no", "cgpa"])
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate students_100k.csv from Kaggle NationalNames.csv."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to NationalNames.csv from Kaggle.",
    )
    parser.add_argument(
        "--output",
        default=Path("data/students_100k.csv"),
        type=Path,
        help="Output CSV path. Default: data/students_100k.csv",
    )
    parser.add_argument(
        "--n",
        default=100000,
        type=int,
        help="Number of output records. Default: 100000",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Deterministic shuffle seed. Default: 42",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = read_name_count_rows(args.input)
    records = generate_records(rows, n=args.n, seed=args.seed)
    write_output(records, args.output)

    print(f"Read valid source rows: {len(rows)}")
    print(f"Wrote benchmark records: {len(records)}")
    print(f"Output: {args.output}")
    print("Schema: name,reg_no,cgpa")
    print("Note: cgpa is synthetic and intended only for benchmarking.")


if __name__ == "__main__":
    main()
