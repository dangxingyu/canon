from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metric(path: Path) -> dict:
    data = json.loads(path.read_text())
    train_loss = data.get("train_loss")
    if train_loss is None:
        train_loss = data.get("train_metrics", {}).get("train_loss")
    eval_loss = data.get("eval_loss")
    if eval_loss is None:
        eval_loss = data.get("eval_metrics", {}).get("eval_loss")
    first_grad = data.get("first_grad_norm")
    if first_grad is None:
        for entry in data.get("log_history", []):
            if "grad_norm" in entry:
                first_grad = entry["grad_norm"]
                break
    runtime = data.get("train_runtime")
    if runtime is None:
        runtime = data.get("train_metrics", {}).get("train_runtime")
    return {
        "path": str(path),
        "canonicalizer": data.get("canonicalizer"),
        "rank": data.get("rank"),
        "target_modules": data.get("target_modules"),
        "seed": data.get("seed"),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "first_grad_norm": first_grad,
        "train_runtime": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Metric files or directories to scan recursively.")
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.rglob("metrics.json")))
        elif path.name == "metrics.json":
            files.append(path)
    rows = [load_metric(path) for path in sorted(set(files))]

    headers = ["rank", "canonicalizer", "target_modules", "seed", "train_loss", "eval_loss", "first_grad_norm", "train_runtime", "path"]
    print("\t".join(headers))
    for row in rows:
        print("\t".join("" if row[h] is None else str(row[h]) for h in headers))


if __name__ == "__main__":
    main()
