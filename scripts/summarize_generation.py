from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean


def discover_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(path.rglob("generation_metrics.json"))
        elif path.name == "generation_metrics.json":
            files.append(path)
    return sorted(set(files))


def infer_run(path: Path) -> tuple[str | None, int | None]:
    match = re.search(r"(original|combined|vo_norm|mlp_norm|actgrad_[^/]+)_seed(\d+)", str(path))
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    rows = []
    for path in discover_files(args.paths):
        data = json.loads(path.read_text())
        canonicalizer, seed = infer_run(path)
        rows.append(
            {
                "canonicalizer": canonicalizer,
                "seed": seed,
                "exact_match": data.get("exact_match"),
                "gold_in_generation": data.get("gold_in_generation"),
                "boxed_rate": data.get("boxed_rate"),
                "num_samples": data.get("num_samples"),
                "path": str(path),
            }
        )

    headers = ["canonicalizer", "seed", "exact_match", "gold_in_generation", "boxed_rate", "num_samples", "path"]
    print("\t".join(headers))
    for row in rows:
        print("\t".join("" if row[key] is None else str(row[key]) for key in headers))

    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["canonicalizer"] is not None:
            groups.setdefault(row["canonicalizer"], []).append(row)

    if groups:
        print("\nsummary")
        print("\t".join(["canonicalizer", "n", "exact_match_mean", "gold_in_generation_mean", "boxed_rate_mean"]))
        for canonicalizer in sorted(groups):
            group = groups[canonicalizer]
            print(
                "\t".join(
                    [
                        canonicalizer,
                        str(len(group)),
                        str(mean(row["exact_match"] for row in group if row["exact_match"] is not None)),
                        str(mean(row["gold_in_generation"] for row in group if row["gold_in_generation"] is not None)),
                        str(mean(row["boxed_rate"] for row in group if row["boxed_rate"] is not None)),
                    ]
                )
            )


if __name__ == "__main__":
    main()
