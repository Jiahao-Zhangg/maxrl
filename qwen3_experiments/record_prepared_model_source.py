#!/usr/bin/env python3
"""Record the immutable source of a locally merged continuation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-step", required=True, type=int)
    args = parser.parse_args()

    weights = sorted(args.model_dir.glob("*.safetensors"))
    if not args.model_dir.joinpath("config.json").is_file() or not weights:
        raise ValueError(f"incomplete Hugging Face model: {args.model_dir}")
    metadata = {
        "source_repo": args.source_repo,
        "source_step": args.source_step,
        "source_checkpoint_folder": f"global_step_{args.source_step}",
        "weight_files": {path.name: path.stat().st_size for path in weights},
    }
    args.model_dir.joinpath("continuation_source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
