import os
import csv
import argparse
from typing import Dict, List, Any, Optional

# 你在这里配置要合并的表格：每个 entry 指向一个 summary_results.csv，并给出 tray 值
# tray 通常就是 stage 数（4-9），但你可以按需要自定义。
INPUT_TABLES = [
    {
        "tray": 4,
        "path": "/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/four_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-44-27/summary_results.csv",
    },
    {
        "tray": 5,
        "path": "/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/five_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-45-05/summary_results.csv",
    },
    {
        "tray": 6,
        "path": "/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/six_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_17-46-07/summary_results.csv",
    },
    {
        "tray": 7,
        "path": "/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/seven_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-54-27/summary_results.csv",
    },
    {
        "tray": 8,
        "path": "/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/eight_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-54-51/summary_results.csv",
    },
    {
        "tray": 9,
        "path": "/home/atom/a/wang7617/SECQUOIA/Pyomos/LD-SDA-Dynamic/results/nine_stage_dynamic_model_switching_nonlinear/nfe30/2026-03-04_18-55-10/summary_results.csv",
    },
]

PREFERRED_COLUMNS = [
    "tray",
    "Algorithm",
    "Subsolver",
    "Norm",
    "Model",
    "Lower Bound",
    "Upper Bound",
    "Gap (%)",
    "Time (s)",
    "Iterations",
    "Status",
    "Termination",
]


def _abs_path(script_dir: str, maybe_rel_path: str) -> str:
    if os.path.isabs(maybe_rel_path):
        return maybe_rel_path
    return os.path.join(script_dir, maybe_rel_path)


def read_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # keep as strings; numeric formatting is already in summary_results.csv
            rows.append(dict(row))
        return rows


def merge_tables(script_dir: str, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    for entry in inputs:
        tray = entry.get("tray", None)
        path = entry.get("path", None)
        if path is None:
            raise ValueError(f"Missing 'path' in entry: {entry}")
        if tray is None:
            raise ValueError(f"Missing 'tray' in entry: {entry}")

        abs_path = _abs_path(script_dir, path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"CSV not found: {abs_path}")

        rows = read_csv_rows(abs_path)
        for row in rows:
            row["tray"] = str(tray)
            merged.append(row)

    return merged


def compute_output_headers(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return PREFERRED_COLUMNS

    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())

    headers: List[str] = []

    # Preferred columns first, if they exist anywhere
    for c in PREFERRED_COLUMNS:
        if c in all_cols:
            headers.append(c)

    # Then any remaining columns (stable, alphabetical)
    remaining = sorted([c for c in all_cols if c not in headers])
    headers.extend(remaining)
    return headers


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    headers = compute_output_headers(rows)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # ensure all headers exist; missing -> empty
            writer.writerow({h: r.get(h, "") for h in headers})


def load_inputs_from_cli(cli_items: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
    """Parse repeated --item arguments of form tray=6,path=..."""
    if not cli_items:
        return None

    inputs: List[Dict[str, Any]] = []
    for item in cli_items:
        # expected: tray=6,path=results/.../summary_results.csv
        entry: Dict[str, Any] = {}
        parts = [p.strip() for p in item.split(",") if p.strip()]
        for p in parts:
            if "=" not in p:
                raise ValueError(f"Invalid --item part (expected key=value): {p}")
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            entry[k] = v
        if "tray" in entry:
            try:
                entry["tray"] = int(entry["tray"])
            except Exception:
                # keep as string if not int
                pass
        inputs.append(entry)
    return inputs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple summary_results.csv tables into one big table, "
            "adding a 'tray' column for each input."
        )
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Default: results/merged_summary_results.csv (relative to script)."
        ),
    )
    parser.add_argument(
        "--item",
        action="append",
        default=None,
        help=(
            "Repeatable. Format: tray=6,path=results/.../summary_results.csv "
            "(path can be absolute or relative to this script)."
        ),
    )

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    inputs = load_inputs_from_cli(args.item)
    if inputs is None:
        inputs = INPUT_TABLES

    if not inputs:
        print("No input tables configured.")
        print("Edit INPUT_TABLES in this script, or pass --item tray=...,path=... multiple times.")
        sys.exit(2)

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(script_dir, "results", "merged_summary_results.csv")
    else:
        out_path = _abs_path(script_dir, out_path)

    merged_rows = merge_tables(script_dir, inputs)
    write_csv(merged_rows, out_path)

    print(f"Merged {len(merged_rows)} rows -> {out_path}")


if __name__ == "__main__":
    import sys

    main()
