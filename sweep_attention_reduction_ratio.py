from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_R_VALUES = (2, 4, 8, 16)

MODELS = (
    ("least_parameters_channel_temporal", "channel_temporal"),
    ("least_parameters_se", "se"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep channel-temporal and SE reduction ratios."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", required=True, choices=("loso", "aos"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-session", default="S001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LEAST PARAMETERS CT + SE REDUCTION-RATIO SWEEP")
    print("=" * 72)
    print(f"Protocol: {args.protocol.upper()}")
    print(f"r values: {list(DEFAULT_R_VALUES)}")
    print(f"Runs per model/r: {args.runs}")
    print(f"Seeds: {args.base_seed}-{args.base_seed + args.runs - 1}")
    print("=" * 72)

    sweep_results = []

    for model_name, model_dir_name in MODELS:
        for r in DEFAULT_R_VALUES:
            r_dir = root / model_dir_name / f"r_{r}"
            ratio_args = (
                ["--channel-reduction-ratio", str(r)]
                if model_name == "least_parameters_channel_temporal"
                else ["--se-reduction-ratio", str(r)]
            )

            command = [
                sys.executable,
                "repeat_experiments.py",
                "--model", model_name,
                "--protocol", args.protocol,
                "--data-root", args.data_root,
                "--output-dir", str(r_dir),
                "--runs", str(args.runs),
                "--base-seed", str(args.base_seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--num-workers", str(args.num_workers),
                "--device", args.device,
                "--calibration-session", args.calibration_session,
                *ratio_args,
            ]

            print("\n" + "=" * 72)
            print(f"MODEL: {model_name}")
            print(f"REDUCTION RATIO: r={r}")
            print(f"OUTPUT: {r_dir}")
            print("=" * 72)

            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)

            summary_path = r_dir / "20_run_summary.json"
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            sweep_results.append(
                {
                    "model": model_name,
                    "r": r,
                    "mean_accuracy_across_runs": summary["mean_accuracy_across_runs"],
                    "std_accuracy_across_runs": summary["std_accuracy_across_runs"],
                    "minimum_run_accuracy": summary["minimum_run_accuracy"],
                    "maximum_run_accuracy": summary["maximum_run_accuracy"],
                    "summary_file": str(summary_path),
                }
            )

            print(
                f"Completed {model_name}, r={r}: "
                f"{summary['mean_accuracy_across_runs'] * 100:.2f}% "
                f"+/- {summary['std_accuracy_across_runs'] * 100:.2f}%"
            )

    final_summary = {
        "models": [name for name, _ in MODELS],
        "protocol": args.protocol,
        "r_values": list(DEFAULT_R_VALUES),
        "runs_per_model_and_r": args.runs,
        "base_seed": args.base_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "results": sweep_results,
    }

    final_path = root / "reduction_ratio_sweep_summary.json"
    with final_path.open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print("\n" + "=" * 72)
    print("FINAL REDUCTION-RATIO SWEEP SUMMARY")
    print("=" * 72)
    for item in sweep_results:
        print(
            f"{item['model']} | r={item['r']} | "
            f"{item['mean_accuracy_across_runs'] * 100:.2f}% "
            f"+/- {item['std_accuracy_across_runs'] * 100:.2f}%"
        )
    print(f"Saved to: {final_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
