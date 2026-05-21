from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canon.model_utils import summarize_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", default="results/architecture/qwen3_0_6b.json")
    args = parser.parse_args()

    summary = summarize_config(args.model_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary.to_dict(), indent=2) + "\n")
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
