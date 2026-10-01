from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, GestureDataset, SampleRecord
from model import build_least_parameters_model, count_trainable_parameters
from model_attention import build_least_parameters_channel_temporal_model
from model_se import build_least_parameters_se_model
from model_eca import build_least_parameters_eca_model
from model_temporal import build_least_parameters_temporal_model
from reproducibility import make_generator, seed_worker, set_global_seed


EXPECTED_PARAMETER_COUNTS = {
    "least_parameters": 1802,
    "least_parameters_eca": 1814,
    "least_parameters_temporal": 1858,
}


def build_selected_model(
    model_name: str,
    device: torch.device,
    channel_reduction_ratio: int,
    se_reduction_ratio: int,
) -> tuple[nn.Module, dict | None]:
    if model_name == "least_parameters":
        model = build_least_parameters_model(in_channels=4, num_classes=6)
        attention = None
    elif model_name == "least_parameters_channel_temporal":
        model = build_least_parameters_channel_temporal_model(
            in_channels=4,
            num_classes=6,
            reduction_ratio=channel_reduction_ratio,
        )
        attention = {
            "type": "channel_temporal",
            "placement": "after_relu_before_dropout",
            "order": "channel_then_temporal",
            "channel_reduction_ratio": channel_reduction_ratio,
            "temporal_kernel_size": 7,
        }
    elif model_name == "least_parameters_temporal":
        model = build_least_parameters_temporal_model(in_channels=4, num_classes=6)
        attention = {
            "type": "temporal",
            "placement": "after_relu_before_dropout",
            "temporal_kernel_size": 7,
        }
    elif model_name == "least_parameters_se":
        model = build_least_parameters_se_model(
            in_channels=4,
            num_classes=6,
            reduction_ratio=se_reduction_ratio,
        )
        attention = {
            "type": "squeeze_excitation",
            "placement": "after_relu_before_dropout",
            "reduction_ratio": se_reduction_ratio,
            "squeeze": "global_average_pool_over_time",
        }
    elif model_name == "least_parameters_eca":
        model = build_least_parameters_eca_model(in_channels=4, num_classes=6)
        attention = {
            "type": "eca_channel",
            "placement": "after_relu_before_dropout",
            "gamma": 2,
            "b": 1,
            "adaptive_kernel_size": 3,
            "squeeze": "global_average_pool_over_time",
            "channel_interaction": "Conv1d across channel descriptor",
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device), attention


@dataclass(frozen=True)
class TrainingConfig:
    channel_reduction_ratio: int
    se_reduction_ratio: int
    model_name: str = "least_parameters"
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()

    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_loader(
    records: Sequence[SampleRecord],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        GestureDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (pred == y).sum().item()
        total_samples += batch_size

        y_cpu = y.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()
        for target, prediction in zip(y_cpu, pred_cpu):
            confusion[target, prediction] += 1

    per_class_precision: dict[str, float | None] = {}
    precision_values: list[float] = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        true_positive = confusion[class_idx, class_idx]
        predicted_as_class = confusion[:, class_idx].sum()

        if predicted_as_class == 0:
            per_class_precision[class_name] = None
        else:
            precision = float(true_positive / predicted_as_class)
            per_class_precision[class_name] = precision
            precision_values.append(precision)

    # Diagnostic only. We keep the same convention as the previous baseline
    # experiments so model comparisons are internally consistent.
    macro_precision = (
        float(np.mean(precision_values)) if precision_values else float("nan")
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_precision_diagnostic": macro_precision,
        "per_class_precision": per_class_precision,
        "confusion_matrix": confusion,
    }


def _save_history(history: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)


def _save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])


def run_fold(
    protocol: str,
    fold_index: int,
    num_folds: int,
    held_out_subject: str,
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    output_dir: str | Path,
    config: TrainingConfig,
    calibration_session: str | None = None,
    calibration_samples: int = 0,
) -> dict:
    """Train one independent LOSO/AOS fold and evaluate only after training."""
    if protocol not in ("loso", "aos"):
        raise ValueError(f"Unknown protocol: {protocol}")
    if protocol == "loso" and (calibration_session is not None or calibration_samples != 0):
        raise ValueError("LOSO must not contain calibration data.")
    if protocol == "aos" and (calibration_session is None or calibration_samples <= 0):
        raise ValueError("AOS requires a calibration session and calibration samples.")

    # Reset RNG before every fold: each fold gets a fresh, reproducible run.
    set_global_seed(config.seed)
    device = select_device(config.device)
    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        train_records,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_records,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=pin_memory,
    )

    # Fresh model and optimizer for every fold.
    model, attention_metadata = build_selected_model(
        config.model_name,
        device,
        channel_reduction_ratio=config.channel_reduction_ratio,
        se_reduction_ratio=config.se_reduction_ratio,
    )
    num_parameters = count_trainable_parameters(model)
    if config.model_name in EXPECTED_PARAMETER_COUNTS:
        expected_parameters = EXPECTED_PARAMETER_COUNTS[config.model_name]
        if num_parameters != expected_parameters:
            raise RuntimeError(
                f"{config.model_name} has {num_parameters} trainable parameters; "
                f"expected {expected_parameters}. Check the architecture."
            )

    criterion = nn.CrossEntropyLoss()

    # The paper specifies AdamW and lr=1e-3 but no explicit weight_decay value.
    # Leaving it unspecified preserves PyTorch's AdamW default rather than
    # inventing another experimental hyperparameter.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("\n" + "=" * 72)
    print(f"Configuration:        {config.model_name}")
    print(f"Protocol:             {protocol.upper()}")
    print(f"Fold:                 {fold_index}/{num_folds}")
    print(f"Held-out subject:     {held_out_subject}")
    print(f"Train source:         train/ only (ds-00 + ds-01)")
    print(f"Test source:          test/ only (canonical)")
    if protocol == "aos":
        print(f"Calibration session:  {calibration_session}")
        print(f"Calibration samples:  {calibration_samples}")
    print(f"Training samples:     {len(train_records)}")
    print(f"Test samples:         {len(test_records)}")
    print(f"Input shape:          (4, 750)")
    print(f"Model:                4 x 16 depthwise-separable CNN")
    print(f"Kernel size:          9")
    print(f"Dropout:              0.2")
    if config.model_name == "least_parameters_channel_temporal":
        print("Attention:            Channel -> Temporal after each ReLU")
        print(f"Channel reduction r:  {config.channel_reduction_ratio}")
        print("Temporal kernel:      7")
    elif config.model_name == "least_parameters_se":
        print("Attention:            SE after each ReLU")
        print(f"SE reduction r:       {config.se_reduction_ratio}")
    elif config.model_name == "least_parameters_temporal":
        print("Attention:            Temporal only after each ReLU")
        print("Temporal kernel:      7")
    elif config.model_name == "least_parameters_eca":
        print("Attention:            ECA channel attention after each ReLU")
        print("ECA gamma, b:         2, 1")
        print("ECA channel kernel:   3 for C=16")
        print("ECA squeeze:          Global average pooling over time")
    print(f"Trainable parameters: {num_parameters}")
    print(f"Device:               {device}")
    print(f"Epochs:               {config.epochs}")
    print(f"Batch size:           {config.batch_size}")
    print(f"Optimizer:            AdamW")
    print(f"Learning rate:        {config.learning_rate}")
    print(f"Seed:                 {config.seed}")
    print("=" * 72)

    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
            }
        )

        if epoch == 1 or epoch == config.epochs or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{config.epochs} | "
                f"train loss {train_loss:.6f} | "
                f"train acc {train_accuracy * 100:.2f}%"
            )

    # No test-set model selection: evaluate only after the fixed 300-epoch run.
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    fold_dir = Path(output_dir).expanduser().resolve() / f"fold_{fold_index:02d}_{held_out_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    _save_history(history, fold_dir / "training_history.csv")
    _save_confusion_matrix(
        test_metrics["confusion_matrix"],
        fold_dir / "confusion_matrix.csv",
    )
    torch.save(model.state_dict(), fold_dir / "model_final.pt")

    result = {
        "configuration_name": config.model_name,
        "protocol": protocol,
        "fold": fold_index,
        "held_out_subject": held_out_subject,
        "calibration_session": calibration_session,
        "calibration_samples": calibration_samples,
        "data_policy": {
            "training_source": "train/",
            "training_variants": ["ds-00", "ds-01"],
            "evaluation_source": "test/",
            "evaluation_variant": "canonical_non_augmented",
            "aos_rule": (
                "other subjects plus held-out calibration session from train/; "
                "held-out non-calibration sessions from test/"
                if protocol == "aos"
                else None
            ),
        },
        "kernel_size": 9,
        "num_blocks": 4,
        "channel_width": 16,
        "widths": [16, 16, 16, 16],
        "dropout": 0.2,
        "attention": attention_metadata,
        "channel_reduction_ratio": config.channel_reduction_ratio,
        "se_reduction_ratio": config.se_reduction_ratio,
        "input_channels": 4,
        "input_samples": 750,
        "num_classes": 6,
        "class_names": list(CLASS_NAMES),
        "num_trainable_parameters": num_parameters,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "final_training_loss": history[-1]["train_loss"],
        "final_training_accuracy": history[-1]["train_accuracy"],
        "final_test_loss": test_metrics["loss"],
        "final_test_accuracy": test_metrics["accuracy"],
        "macro_precision_diagnostic": test_metrics["macro_precision_diagnostic"],
        "per_class_precision": test_metrics["per_class_precision"],
        "seed": config.seed,
        "epoch_count": config.epochs,
        "batch_size": config.batch_size,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "loss": "CrossEntropyLoss",
        "device": str(device),
    }

    with (fold_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("-" * 72)
    print(f"{protocol.upper()} Fold {fold_index} Results | {held_out_subject}")
    print(f"Final Train Loss:     {result['final_training_loss']:.6f}")
    print(f"Final Train Accuracy: {result['final_training_accuracy'] * 100:.2f}%")
    print(f"Final Test Loss:      {result['final_test_loss']:.6f}")
    print(f"Final Test Accuracy:  {result['final_test_accuracy'] * 100:.2f}%")
    print(
        "Macro Precision*:     "
        f"{result['macro_precision_diagnostic'] * 100:.2f}%"
    )
    print("* diagnostic; kept consistent with the previous experiments")
    print("-" * 72)

    return result
