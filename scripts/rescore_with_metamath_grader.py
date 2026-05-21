from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "metamath"))
import util  # noqa: E402


def remove_boxed(s: str | None) -> str | None:
    if s is None:
        return None
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def metamath_extract(generation: str) -> str | None:
    boxed = util.last_boxed_only_string(generation)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def rescore_file(predictions_path: Path) -> dict[str, object]:
    records: list[dict] = []
    with predictions_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n = len(records)
    strict_correct = 0
    strict_extracted = 0
    hybrid_correct = 0
    by_subject_strict: dict[str, list[int]] = defaultdict(list)
    by_level_strict: dict[str, list[int]] = defaultdict(list)
    rescored: list[dict] = []

    for record in records:
        gold = record["gold_answer"]
        generation = record["generation"]
        pred_existing = record.get("pred_answer", "") or ""

        meta_pred = metamath_extract(generation)
        strict_match = bool(meta_pred is not None and util.is_equiv(meta_pred, gold))
        if meta_pred is not None:
            strict_extracted += 1
        if strict_match:
            strict_correct += 1
        hybrid_match = bool(util.is_equiv(pred_existing, gold))
        if hybrid_match:
            hybrid_correct += 1

        by_subject_strict[str(record.get("subject"))].append(int(strict_match))
        by_level_strict[str(record.get("level"))].append(int(strict_match))

        rescored.append({
            **record,
            "metamath_pred": meta_pred,
            "metamath_strict_match": strict_match,
            "metamath_hybrid_match": hybrid_match,
        })

    summary = {
        "num_samples": n,
        "strict_exact_match": strict_correct / n if n else 0.0,
        "strict_extraction_rate": strict_extracted / n if n else 0.0,
        "hybrid_exact_match": hybrid_correct / n if n else 0.0,
        "num_strict_correct": strict_correct,
        "num_strict_extracted": strict_extracted,
        "num_hybrid_correct": hybrid_correct,
        "by_subject_strict": {
            k: {"n": len(v), "exact_match": sum(v) / len(v) if v else 0.0}
            for k, v in sorted(by_subject_strict.items())
        },
        "by_level_strict": {
            k: {"n": len(v), "exact_match": sum(v) / len(v) if v else 0.0}
            for k, v in sorted(by_level_strict.items())
        },
    }
    return {"summary": summary, "records": rescored}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        default="results/math_generation_eval/qwen3_0_6b/lr2e-5_steps1000_math500",
        help="Directory containing one subdir per run, each with predictions.jsonl",
    )
    parser.add_argument(
        "--out-name",
        default="metamath_rescore.json",
        help="Filename written next to each predictions.jsonl",
    )
    parser.add_argument(
        "--also-write-records",
        action="store_true",
        help="If set, also write metamath_rescored.jsonl beside the original predictions",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise SystemExit(f"runs root not found: {runs_root}")

    aggregate: dict[str, dict] = {}
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        predictions = run_dir / "predictions.jsonl"
        if not predictions.exists():
            continue
        result = rescore_file(predictions)
        (run_dir / args.out_name).write_text(json.dumps(result["summary"], indent=2) + "\n")
        if args.also_write_records:
            with (run_dir / "metamath_rescored.jsonl").open("w") as handle:
                for rec in result["records"]:
                    handle.write(json.dumps(rec) + "\n")
        aggregate[run_dir.name] = result["summary"]

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
