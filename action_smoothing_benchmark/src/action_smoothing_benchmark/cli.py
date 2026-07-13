from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import scipy

matplotlib.use("Agg")

from . import __version__
from .data import load_action_frames, make_chunks, validate_frames
from .methods import build_method_catalog
from .metrics import SCORE_WEIGHTS, run_benchmark
from .report import write_report

DEFAULT_DATASET = Path(r"C:\Users\ccu\.cache\huggingface\lerobot\wuc1\rollout_bi_so101_ffp_0615-14-12-dagger_merged3cam_nopolicyaction_no_use_smoothing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare action-chunk smoothing methods on a LeRobot v3 dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("results_smoothing_first"))
    parser.add_argument("--chunk-length", type=int, default=50)
    parser.add_argument("--fps", type=float, default=None, help="Override the FPS in meta/info.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading action data from {args.dataset_root}")
    info, frames = load_action_frames(args.dataset_root.resolve())
    warnings = validate_frames(frames)
    fps = float(args.fps) if args.fps is not None else info.fps
    if fps != info.fps:
        info = type(info)(info.root, fps, info.action_names, info.total_frames, info.total_episodes)
    chunks = make_chunks(frames, args.chunk_length)
    methods = build_method_catalog()
    print(f"Running {len(methods)} variants on {len(chunks)} reconstructed chunks")
    result = run_benchmark(chunks, methods, info.action_names, fps)

    result.per_chunk.to_csv(output_dir / "per_chunk.csv", index=False)
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "errors.json").write_text(json.dumps(result.errors, indent=2, ensure_ascii=False), encoding="utf-8")
    run_config = {
        "benchmark_version": __version__,
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_root": str(info.root),
        "dataset_note": "Chunks are reconstructed from frame-level action and are not original network chunk boundaries.",
        "fps": fps,
        "chunk_length": args.chunk_length,
        "total_frames": info.total_frames,
        "total_episodes": info.total_episodes,
        "total_chunks": len(chunks),
        "partial_chunks": sum(chunk.is_partial for chunk in chunks),
        "action_names": info.action_names,
        "methods": [{"method_id": method.method_id, "family": method.family, "causal": method.causal, "parameters": method.parameters} for method in methods],
        "score_weights": SCORE_WEIGHTS,
        "warnings": warnings,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir, info, chunks, result, warnings)
    winner = result.summary[result.summary["method_id"] != "raw"].iloc[0]
    print(f"Best non-raw method: {winner['method_id']} (score={winner['deployment_score']:.2f})")
    print(f"Report: {output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
