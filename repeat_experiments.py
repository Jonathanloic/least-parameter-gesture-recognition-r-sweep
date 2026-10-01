from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


MODEL_CHOICES = (
    "least_parameters",
    "least_parameters_channel_temporal",
    "least_parameters_temporal",
    "least_parameters_se",
    "least_parameters_eca",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat a complete Least Parameters LOSO/AOS experiment multiple "
            "times with different seeds and summarize the run-level accuracies."
        )
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--protocol", required=True, choices=("loso", "aos"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channel-reduction-ratio", type=int, required=True)
    parser.add_argument("--se-reduction-ratio", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-session", default="S001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_results: list[dict] = []

    for run_index in range(args.runs):
        run_number = run_index + 1
        seed = args.base_seed + run_index
        run_dir = output_root / f"run_{run_number:02d}"

        print("\n" + "=" * 72)
        print(f"RUN {run_number}/{args.runs}")
        print(f"Model:    {args.model}")
        print(f"Protocol: {args.protocol.upper()}")
        print(f"Seed:     {seed}")
        print("=" * 72)

        command = [
            sys.executable,
            "experiments.py",
            "--model", args.model,
            "--data-root", args.data_root,
            "--protocol", args.protocol,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--channel-reduction-ratio", str(args.channel_reduction_ratio),
            "--se-reduction-ratio", str(args.se_reduction_ratio),
            "--seed", str(seed),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--output-dir", str(run_dir),
        ]

        if args.protocol == "aos":
            command.extend(["--calibration-session", args.calibration_session])

        subprocess.run(command, check=True)

        summary_path = run_dir / args.protocol / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Expected summary not found after run {run_number}: {summary_path}"
            )

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        run_accuracy = float(summary["mean_test_accuracy"])
        run_results.append(
            {
                "run": run_number,
                "seed": seed,
                "mean_test_accuracy": run_accuracy,
                "summary_file": str(summary_path),
            }
        )

        print(
            f"Run {run_number:02d} finished: "
            f"{run_accuracy * 100:.2f}% mean fold accuracy"
        )

    accuracies = [r["mean_test_accuracy"] for r in run_results]

    final_summary = {
        "model": args.model,
        "protocol": args.protocol,
        "num_runs": args.runs,
        "base_seed": args.base_seed,
        "seeds": [args.base_seed + i for i in range(args.runs)],
        "epochs_per_fold": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "channel_reduction_ratio": args.channel_reduction_ratio,
        "se_reduction_ratio": args.se_reduction_ratio,
        "mean_accuracy_across_runs": mean(accuracies),
        "std_accuracy_across_runs": pstdev(accuracies),
        "minimum_run_accuracy": min(accuracies),
        "maximum_run_accuracy": max(accuracies),
        "runs": run_results,
    }

    output_file = output_root / "20_run_summary.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    print("\n" + "=" * 72)
    print("FINAL REPEATED-RUN RESULT")
    print("=" * 72)
    print(f"Model:    {args.model}")
    print(f"Protocol: {args.protocol.upper()}")
    print(f"Runs:     {args.runs}")
    print(
        f"Accuracy: {final_summary['mean_accuracy_across_runs'] * 100:.2f}% "
        f"± {final_summary['std_accuracy_across_runs'] * 100:.2f}%"
    )
    print(f"Minimum:  {final_summary['minimum_run_accuracy'] * 100:.2f}%")
    print(f"Maximum:  {final_summary['maximum_run_accuracy'] * 100:.2f}%")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
