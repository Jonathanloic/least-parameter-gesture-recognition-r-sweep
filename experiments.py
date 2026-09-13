from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from dataset import (
    dataset_summary,
    make_aos_split,
    make_loso_split,
    scan_dataset,
    subjects,
    validate_augmentation_structure,
    validate_sample_arrays,
)
from train import TrainingConfig, run_fold


DEFAULT_CALIBRATION_SESSION = "S001"

LEAST_PARAMETERS_CONFIG = {
    "name": "least_parameters",
    "preprocessed_data": {
        "band_pass_hz": [225, 375],
        "downsampling_ratio": 2,
        "window_ms": 1500,
        "saved_input_shape": [4, 750],
        "preprocessing_policy": "already applied; loader must not reapply it",
        "training_data": "train/ contains ds-00 and ds-01 variants",
        "evaluation_data": "test/ contains one canonical non-augmented sample per event",
    },
    "model": {
        "kernel_size": 9,
        "num_blocks": 4,
        "channel_width": 16,
        "widths": [16, 16, 16, 16],
        "dropout": 0.2,
        "input_channels": 4,
        "num_classes": 6,
    },
    "protocols": {
        "loso": {
            "training": "all non-held-out subjects from train/",
            "evaluation": "held-out subject from test/",
        },
        "aos": {
            "training": (
                "all non-held-out subjects from train/ plus held-out subject "
                "calibration session from train/"
            ),
            "evaluation": (
                "held-out subject non-calibration sessions from test/"
            ),
            "default_calibration_session": DEFAULT_CALIBRATION_SESSION,
        },
    },
}


LEAST_PARAMETERS_CHANNEL_TEMPORAL_CONFIG = {
    **LEAST_PARAMETERS_CONFIG,
    "name": "least_parameters_channel_temporal",
    "model": {
        **LEAST_PARAMETERS_CONFIG["model"],
        "attention": {
            "type": "channel_temporal",
            "placement": "after_relu_before_dropout",
            "order": "channel_then_temporal",
            "temporal_kernel_size": 7,
        },
    },
}

LEAST_PARAMETERS_SE_CONFIG = {
    **LEAST_PARAMETERS_CONFIG,
    "name": "least_parameters_se",
    "model": {
        **LEAST_PARAMETERS_CONFIG["model"],
        "attention": {
            "type": "squeeze_excitation",
            "placement": "after_relu_before_dropout",
            "squeeze": "global_average_pool_over_time",
        },
    },
}

LEAST_PARAMETERS_ECA_CONFIG = {
    **LEAST_PARAMETERS_CONFIG,
    "name": "least_parameters_eca",
    "model": {
        **LEAST_PARAMETERS_CONFIG["model"],
        "attention": {
            "type": "eca_channel",
            "placement": "after_relu_before_dropout",
            "gamma": 2,
            "b": 1,
            "adaptive_kernel_size_for_C16": 3,
            "squeeze": "global_average_pool_over_time",
            "channel_interaction": "Conv1d across channel descriptor",
        },
    },
}

MODEL_CONFIGS = {
    "least_parameters": LEAST_PARAMETERS_CONFIG,
    "least_parameters_channel_temporal": LEAST_PARAMETERS_CHANNEL_TEMPORAL_CONFIG,
    "least_parameters_se": LEAST_PARAMETERS_SE_CONFIG,
    "least_parameters_eca": LEAST_PARAMETERS_ECA_CONFIG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Least Parameters 1D-SepCNN with LOSO and AOS. Training/calibration "
            "always comes from augmented train/ data; evaluation always comes "
            "from canonical test/ data."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to the extracted least-parameter directory containing train/ and test/.",
    )
    parser.add_argument(
        "--model",
        choices=(
            "least_parameters",
            "least_parameters_channel_temporal",
            "least_parameters_se",
            "least_parameters_eca",
        ),
        default="least_parameters",
        help="Model variant to train. Baseline is the default.",
    )
    parser.add_argument(
        "--protocol",
        choices=("loso", "aos"),
        default="loso",
        help="Evaluation protocol: LOSO or AOS.",
    )
    parser.add_argument(
        "--calibration-session",
        default=DEFAULT_CALIBRATION_SESSION,
        help=(
            "Held-out participant session used for AOS calibration. "
            f"Default: {DEFAULT_CALIBRATION_SESSION}. Ignored for LOSO."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channel-reduction-ratio", type=int, required=True)
    parser.add_argument("--se-reduction-ratio", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, ...",
    )
    parser.add_argument(
        "--output-dir",
        default="results_least_parameters",
        help="Directory where fold histories, models and summaries are saved.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help=(
            "Optional held-out participant subset (for example P001) for a smoke test. "
            "Omit for the full 15-fold evaluation."
        ),
    )
    parser.add_argument(
        "--validate-all-samples",
        action="store_true",
        help="Open every .npz at startup and validate shape/dtype/finiteness.",
    )
    return parser.parse_args()


def save_protocol_summary(results: list[dict], protocol_dir: Path) -> dict:
    accuracies = [float(result["final_test_accuracy"]) for result in results]
    macro_precisions = [
        float(result["macro_precision_diagnostic"]) for result in results
    ]

    summary = {
        "configuration_name": results[0]["configuration_name"],
        "protocol": results[0]["protocol"],
        "num_folds": len(results),
        "mean_test_accuracy": mean(accuracies),
        "std_test_accuracy": pstdev(accuracies),
        "mean_macro_precision_diagnostic": mean(macro_precisions),
        "std_macro_precision_diagnostic": pstdev(macro_precisions),
        "folds": results,
    }

    with (protocol_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (protocol_dir / "fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "configuration_name",
                "fold",
                "held_out_subject",
                "calibration_session",
                "calibration_samples",
                "train_samples",
                "test_samples",
                "final_training_loss",
                "final_training_accuracy",
                "final_test_loss",
                "final_test_accuracy",
                "macro_precision_diagnostic",
                "num_trainable_parameters",
                "seed",
                "epoch_count",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})

    return summary


def run_protocol(
    protocol: str,
    source_train_records,
    source_test_records,
    fold_subjects: list[str],
    output_root: Path,
    training_config: TrainingConfig,
    calibration_session: str = DEFAULT_CALIBRATION_SESSION,
) -> dict:
    protocol_dir = output_root / protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for fold_index, held_out_subject in enumerate(fold_subjects, start=1):
        if protocol == "loso":
            fold_train_records, fold_test_records = make_loso_split(
                source_train_records,
                source_test_records,
                held_out_subject=held_out_subject,
            )
            fold_calibration_session = None
            calibration_samples = 0

        elif protocol == "aos":
            fold_train_records, fold_test_records = make_aos_split(
                source_train_records,
                source_test_records,
                held_out_subject=held_out_subject,
                calibration_session=calibration_session,
            )
            fold_calibration_session = calibration_session
            calibration_samples = sum(
                1
                for record in fold_train_records
                if (
                    record.subject == held_out_subject
                    and record.session == calibration_session
                )
            )

        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        result = run_fold(
            protocol=protocol,
            fold_index=fold_index,
            num_folds=len(fold_subjects),
            held_out_subject=held_out_subject,
            train_records=fold_train_records,
            test_records=fold_test_records,
            output_dir=protocol_dir,
            config=training_config,
            calibration_session=fold_calibration_session,
            calibration_samples=calibration_samples,
        )
        results.append(result)

    summary = save_protocol_summary(results, protocol_dir)

    print("\n" + "=" * 72)
    print(f"FINAL {protocol.upper()} RESULTS")
    print("=" * 72)
    for result in results:
        print(
            f"{result['held_out_subject']}: "
            f"Accuracy = {result['final_test_accuracy'] * 100:.2f}% | "
            f"Macro precision* = "
            f"{result['macro_precision_diagnostic'] * 100:.2f}%"
        )
    print("-" * 72)
    print(f"Mean Accuracy:         {summary['mean_test_accuracy'] * 100:.2f}%")
    print(f"Std Accuracy:          {summary['std_test_accuracy'] * 100:.2f}%")
    print(
        "Mean Macro Precision*: "
        f"{summary['mean_macro_precision_diagnostic'] * 100:.2f}%"
    )
    print(
        "Std Macro Precision*:  "
        f"{summary['std_macro_precision_diagnostic'] * 100:.2f}%"
    )
    print("* diagnostic; kept consistent with the previous experiments")
    print("=" * 72)

    return summary


def main() -> None:
    args = parse_args()

    if args.channel_reduction_ratio <= 0 or args.se_reduction_ratio <= 0:
        raise ValueError("Reduction ratios must be positive integers.")

    source_train_records, source_test_records = scan_dataset(args.data_root)

    # Validate the crucial Least Parameters organization before any training.
    validate_augmentation_structure(source_train_records, source_test_records)
    validate_sample_arrays(
        source_train_records,
        source_test_records,
        check_all=args.validate_all_samples,
    )
    summary = dataset_summary(source_train_records, source_test_records)

    print(f"Selected model:   {args.model}")
    print("Dataset detected:")
    print(f"  train/ samples: {summary['train_samples']}")
    print(f"  test/ samples:  {summary['test_samples']}")
    print(f"  original events:{summary['original_events']}")
    print(f"  subjects:       {summary['num_subjects']} ({', '.join(summary['subjects'])})")
    print(f"  sessions:       {summary['num_sessions']} ({', '.join(summary['sessions'])})")
    print(f"  classes:        {summary['num_classes']} ({', '.join(summary['class_names'])})")
    print(f"  input shape:    {tuple(summary['input_shape'])}")
    print(f"  train variants: {summary['train_downsampling_variant_counts']}")
    print(f"  train classes:  {summary['train_samples_per_gesture']}")
    print(f"  test classes:   {summary['test_samples_per_gesture']}")
    print("  preprocessing:  already applied; loader does not modify signal")

    if args.protocol == "loso":
        print("  LOSO policy:    other subjects from train/; held-out subject from test/")
    else:
        print(
            "  AOS policy:     other subjects + held-out calibration session from train/; "
            "held-out non-calibration sessions from test/"
        )
        print(f"  calibration:    {args.calibration_session}")

    all_train_subjects = subjects(source_train_records)
    all_test_subjects = subjects(source_test_records)
    if all_train_subjects != all_test_subjects:
        raise RuntimeError(
            "train/ and test/ subject sets differ: "
            f"train={all_train_subjects}, test={all_test_subjects}"
        )

    all_subjects = all_train_subjects

    if args.subjects:
        requested = args.subjects
        unknown = sorted(set(requested) - set(all_subjects))
        if unknown:
            raise ValueError(f"Unknown subject(s): {unknown}")
        fold_subjects = [subject for subject in all_subjects if subject in requested]
        print(
            "WARNING: --subjects was supplied, so this is a partial/smoke run, "
            f"not the full 15-fold {args.protocol.upper()} evaluation."
        )
    else:
        fold_subjects = all_subjects
        if len(fold_subjects) != 15:
            raise RuntimeError(
                f"Full evaluation expects 15 subjects; found {len(fold_subjects)}."
            )

    training_config = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        channel_reduction_ratio=args.channel_reduction_ratio,
        se_reduction_ratio=args.se_reduction_ratio,
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_configuration = {
        **MODEL_CONFIGS[args.model],
        "model": {**MODEL_CONFIGS[args.model]["model"]},
    }
    if args.model == "least_parameters_channel_temporal":
        selected_configuration["model"]["attention"] = {
            **selected_configuration["model"]["attention"],
            "channel_reduction_ratio": args.channel_reduction_ratio,
        }
    elif args.model == "least_parameters_se":
        selected_configuration["model"]["attention"] = {
            **selected_configuration["model"]["attention"],
            "reduction_ratio": args.se_reduction_ratio,
        }

    with (output_root / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "configuration": selected_configuration,
                "selected_model": args.model,
                "implemented_models": list(MODEL_CONFIGS),
                "training": {
                    "epochs": training_config.epochs,
                    "batch_size": training_config.batch_size,
                    "learning_rate": training_config.learning_rate,
                    "seed": training_config.seed,
                    "num_workers": training_config.num_workers,
                    "device": training_config.device,
                    "channel_reduction_ratio": training_config.channel_reduction_ratio,
                    "se_reduction_ratio": training_config.se_reduction_ratio,
                },
                "dataset": summary,
                "fold_subjects": fold_subjects,
                "selected_protocol": args.protocol,
                "implemented_protocols": ["loso", "aos"],
                "aos_calibration_session": (
                    args.calibration_session if args.protocol == "aos" else None
                ),
            },
            handle,
            indent=2,
        )

    run_protocol(
        protocol=args.protocol,
        source_train_records=source_train_records,
        source_test_records=source_test_records,
        fold_subjects=fold_subjects,
        output_root=output_root,
        training_config=training_config,
        calibration_session=args.calibration_session,
    )


if __name__ == "__main__":
    main()
