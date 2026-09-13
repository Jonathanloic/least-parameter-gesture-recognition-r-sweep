from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import re

import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = (
    "knock",
    "swipe-down",
    "swipe-left",
    "swipe-right",
    "swipe-up",
    "tap",
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

EXPECTED_INPUT_CHANNELS = 4
EXPECTED_WINDOW_SAMPLES = 750
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

_FILENAME_RE = re.compile(r"_event-(?P<event>\d+)(?:_ds-(?P<variant>\d+))?$")


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    source_split: str
    subject: str
    session: str
    gesture: str
    event: str
    downsampling_variant: str | None

    @property
    def event_key(self) -> tuple[str, str, str, str]:
        """Identity of the original recorded event, independent of augmentation."""
        return (self.subject, self.session, self.gesture, self.event)


def _parse_filename(path: Path) -> tuple[str, str | None]:
    match = _FILENAME_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse event/variant from filename: {path.name}")
    return match.group("event"), match.group("variant")


def scan_split(data_root: str | Path, source_split: str) -> list[SampleRecord]:
    """Scan one preprocessed Least Parameters split.

    Expected layout:
        data_root/train/sub-PXXX/ses-SXXX/<gesture>/*.npz
        data_root/test/sub-PXXX/ses-SXXX/<gesture>/*.npz

    Signal preprocessing is NOT performed here. The saved arrays are already
    band-pass filtered, windowed, and downsampled.
    """
    if source_split not in (TRAIN_SPLIT, TEST_SPLIT):
        raise ValueError(f"Unknown source split: {source_split}")

    root = Path(data_root).expanduser().resolve()
    split_root = root / source_split

    if not split_root.exists():
        raise FileNotFoundError(f"Missing dataset split directory: {split_root}")
    if not split_root.is_dir():
        raise NotADirectoryError(f"Dataset split is not a directory: {split_root}")

    records: list[SampleRecord] = []

    for path in sorted(split_root.glob("sub-P*/ses-S*/*/*.npz")):
        # Ignore macOS AppleDouble metadata if present after extraction/copying.
        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue

        gesture = path.parent.name
        session_dir = path.parent.parent.name
        subject_dir = path.parent.parent.parent.name

        if gesture not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unexpected gesture directory '{gesture}' in {path}. "
                f"Expected one of {CLASS_NAMES}."
            )
        if not subject_dir.startswith("sub-"):
            raise ValueError(f"Unexpected subject directory: {subject_dir}")
        if not session_dir.startswith("ses-"):
            raise ValueError(f"Unexpected session directory: {session_dir}")

        event, variant = _parse_filename(path)

        if source_split == TRAIN_SPLIT and variant not in ("00", "01"):
            raise ValueError(
                f"Training sample must be ds-00 or ds-01, got {path.name}"
            )
        if source_split == TEST_SPLIT and variant is not None:
            raise ValueError(
                f"Test sample must be canonical/non-augmented, got {path.name}"
            )

        records.append(
            SampleRecord(
                path=path,
                source_split=source_split,
                subject=subject_dir.removeprefix("sub-"),
                session=session_dir.removeprefix("ses-"),
                gesture=gesture,
                event=event,
                downsampling_variant=variant,
            )
        )

    if not records:
        raise RuntimeError(
            f"No real .npz samples found under {split_root}. "
            "Expected sub-P*/ses-S*/*/*.npz."
        )

    return records


def scan_dataset(
    data_root: str | Path,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Return the separate augmented-training and canonical-test record lists."""
    train_records = scan_split(data_root, TRAIN_SPLIT)
    test_records = scan_split(data_root, TEST_SPLIT)
    return train_records, test_records


def validate_augmentation_structure(
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
) -> None:
    """Validate the intended ds-00/ds-01 augmentation organization.

    For every original event:
      * train/ must contain both ds-00 and ds-01
      * test/ must contain one canonical version
      * train and test must describe the same set of original events

    This validates metadata only; it does not alter or compare signal values.
    """
    train_variants: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for record in train_records:
        if record.source_split != TRAIN_SPLIT:
            raise AssertionError("Non-training record found in train_records.")
        if record.downsampling_variant is None:
            raise AssertionError(f"Missing training augmentation tag: {record.path}")
        train_variants[record.event_key].add(record.downsampling_variant)

    bad_variant_keys = {
        key: variants
        for key, variants in train_variants.items()
        if variants != {"00", "01"}
    }
    if bad_variant_keys:
        example_key, variants = next(iter(bad_variant_keys.items()))
        raise RuntimeError(
            "Training augmentation structure is incomplete. "
            f"Example event {example_key} has variants {sorted(variants)}; "
            "expected ['00', '01']."
        )

    test_keys: list[tuple[str, str, str, str]] = []
    for record in test_records:
        if record.source_split != TEST_SPLIT:
            raise AssertionError("Non-test record found in test_records.")
        if record.downsampling_variant is not None:
            raise AssertionError(f"Augmented sample found in test set: {record.path}")
        test_keys.append(record.event_key)

    if len(test_keys) != len(set(test_keys)):
        raise RuntimeError("Duplicate canonical event found in test/.")

    train_keys = set(train_variants)
    canonical_keys = set(test_keys)
    if train_keys != canonical_keys:
        only_train = sorted(train_keys - canonical_keys)[:5]
        only_test = sorted(canonical_keys - train_keys)[:5]
        raise RuntimeError(
            "train/ and test/ do not represent the same original events. "
            f"Only in train (first 5): {only_train}; "
            f"only in test (first 5): {only_test}."
        )


def validate_sample_arrays(
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    check_all: bool = True,
) -> None:
    """Validate stored model-ready arrays without changing the signal."""
    records = list(train_records) + list(test_records)
    to_check = records if check_all else records[:1] + records[len(train_records):len(train_records)+1]

    for record in to_check:
        with np.load(record.path, allow_pickle=False) as data:
            if "x" not in data.files:
                raise KeyError(f"Missing array key 'x' in {record.path}")
            x = data["x"]

        if x.shape != (EXPECTED_INPUT_CHANNELS, EXPECTED_WINDOW_SAMPLES):
            raise ValueError(
                f"Unexpected shape {x.shape} in {record.path}; expected "
                f"({EXPECTED_INPUT_CHANNELS}, {EXPECTED_WINDOW_SAMPLES})."
            )
        if x.dtype != np.float32:
            raise TypeError(
                f"Unexpected dtype {x.dtype} in {record.path}; expected float32."
            )
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/Inf detected in {record.path}")


def subjects(records: Iterable[SampleRecord]) -> list[str]:
    return sorted({record.subject for record in records})


def dataset_summary(
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
) -> dict:
    train_by_subject = Counter(record.subject for record in train_records)
    test_by_subject = Counter(record.subject for record in test_records)
    train_by_gesture = Counter(record.gesture for record in train_records)
    test_by_gesture = Counter(record.gesture for record in test_records)

    train_subjects = subjects(train_records)
    test_subjects = subjects(test_records)
    if train_subjects != test_subjects:
        raise RuntimeError(
            f"Subject mismatch between train/ and test/: "
            f"train={train_subjects}, test={test_subjects}"
        )

    sessions = sorted(
        {record.session for record in train_records}
        | {record.session for record in test_records}
    )

    variant_counts = Counter(
        record.downsampling_variant for record in train_records
    )

    return {
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "original_events": len(test_records),
        "subjects": train_subjects,
        "num_subjects": len(train_subjects),
        "sessions": sessions,
        "num_sessions": len(sessions),
        "class_names": list(CLASS_NAMES),
        "num_classes": len(CLASS_NAMES),
        "train_samples_per_subject": dict(sorted(train_by_subject.items())),
        "test_samples_per_subject": dict(sorted(test_by_subject.items())),
        "train_samples_per_gesture": {
            name: train_by_gesture[name] for name in CLASS_NAMES
        },
        "test_samples_per_gesture": {
            name: test_by_gesture[name] for name in CLASS_NAMES
        },
        "train_downsampling_variant_counts": {
            "ds-00": variant_counts["00"],
            "ds-01": variant_counts["01"],
        },
        "input_shape": [EXPECTED_INPUT_CHANNELS, EXPECTED_WINDOW_SAMPLES],
    }


def make_loso_split(
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    held_out_subject: str,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Least Parameters LOSO with strict train/test-root separation.

    Training:
        All subjects except held_out_subject, loaded ONLY from train/.
        This keeps both ds-00 and ds-01 training augmentations.

    Evaluation:
        held_out_subject loaded ONLY from test/.
        Therefore evaluation uses only the canonical non-augmented event version.
    """
    known_subjects = set(subjects(train_records)) | set(subjects(test_records))
    if held_out_subject not in known_subjects:
        raise ValueError(f"Unknown held-out subject: {held_out_subject}")

    fold_train = [
        record
        for record in train_records
        if record.subject != held_out_subject
    ]
    fold_test = [
        record
        for record in test_records
        if record.subject == held_out_subject
    ]

    if not fold_train:
        raise RuntimeError(f"LOSO training set is empty for {held_out_subject}.")
    if not fold_test:
        raise RuntimeError(f"LOSO test set is empty for {held_out_subject}.")

    # Leakage/split guards.
    if any(record.source_split != TRAIN_SPLIT for record in fold_train):
        raise AssertionError("LOSO training data must come only from train/.")
    if any(record.source_split != TEST_SPLIT for record in fold_test):
        raise AssertionError("LOSO evaluation data must come only from test/.")
    if any(record.subject == held_out_subject for record in fold_train):
        raise AssertionError("LOSO leakage: held-out subject found in training data.")
    if any(record.subject != held_out_subject for record in fold_test):
        raise AssertionError("LOSO split error: evaluation contains another subject.")
    if any(record.downsampling_variant not in ("00", "01") for record in fold_train):
        raise AssertionError("LOSO training sample lacks expected ds-00/ds-01 tag.")
    if any(record.downsampling_variant is not None for record in fold_test):
        raise AssertionError("LOSO evaluation contains an augmented variant.")

    return fold_train, fold_test


def make_aos_split(
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    held_out_subject: str,
    calibration_session: str = "S001",
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Least Parameters AOS with strict train/test-root separation.

    Training:
        * all other subjects, loaded ONLY from train/; and
        * the held-out subject's calibration session (default S001), also loaded
          ONLY from train/. Both ds-00 and ds-01 are retained for calibration.

    Evaluation:
        * only the held-out subject's non-calibration sessions, loaded ONLY from
          test/. Therefore evaluation remains canonical/non-augmented.

    Example for P001 with calibration_session='S001':
        train = train/P002..P015 + train/P001/S001
        test  = test/P001/S002..S010
    """
    known_subjects = set(subjects(train_records)) | set(subjects(test_records))
    if held_out_subject not in known_subjects:
        raise ValueError(f"Unknown held-out subject: {held_out_subject}")

    known_sessions = {record.session for record in list(train_records) + list(test_records)}
    if calibration_session not in known_sessions:
        raise ValueError(
            f"Unknown calibration session: {calibration_session}; "
            f"known sessions are {sorted(known_sessions)}"
        )

    fold_train = [
        record
        for record in train_records
        if (
            record.subject != held_out_subject
            or (
                record.subject == held_out_subject
                and record.session == calibration_session
            )
        )
    ]
    fold_test = [
        record
        for record in test_records
        if (
            record.subject == held_out_subject
            and record.session != calibration_session
        )
    ]

    if not fold_train:
        raise RuntimeError(f"AOS training set is empty for {held_out_subject}.")
    if not fold_test:
        raise RuntimeError(f"AOS test set is empty for {held_out_subject}.")

    calibration_records = [
        record
        for record in fold_train
        if (
            record.subject == held_out_subject
            and record.session == calibration_session
        )
    ]
    if not calibration_records:
        raise RuntimeError(
            f"No AOS calibration samples found for {held_out_subject}/"
            f"{calibration_session}."
        )

    # Leakage/split guards.
    if any(record.source_split != TRAIN_SPLIT for record in fold_train):
        raise AssertionError("AOS training data must come only from train/.")
    if any(record.source_split != TEST_SPLIT for record in fold_test):
        raise AssertionError("AOS evaluation data must come only from test/.")
    if any(
        record.subject == held_out_subject
        and record.session != calibration_session
        for record in fold_train
    ):
        raise AssertionError(
            "AOS leakage: held-out subject non-calibration session found in training."
        )
    if any(record.subject != held_out_subject for record in fold_test):
        raise AssertionError("AOS split error: evaluation contains another subject.")
    if any(record.session == calibration_session for record in fold_test):
        raise AssertionError(
            "AOS leakage: calibration session found in the evaluation set."
        )
    if any(record.downsampling_variant not in ("00", "01") for record in fold_train):
        raise AssertionError("AOS training sample lacks expected ds-00/ds-01 tag.")
    if any(record.downsampling_variant is not None for record in fold_test):
        raise AssertionError("AOS evaluation contains an augmented variant.")

    # Every calibration event in train/ must contribute both downsampling phases.
    calibration_variants: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for record in calibration_records:
        if record.downsampling_variant is None:
            raise AssertionError(f"Missing AOS calibration variant tag: {record.path}")
        calibration_variants[record.event_key].add(record.downsampling_variant)

    incomplete = {
        key: variants
        for key, variants in calibration_variants.items()
        if variants != {"00", "01"}
    }
    if incomplete:
        key, variants = next(iter(incomplete.items()))
        raise RuntimeError(
            f"Incomplete AOS calibration augmentation for {key}: "
            f"found {sorted(variants)}, expected ['00', '01']."
        )

    return fold_train, fold_test


class GestureDataset(Dataset):
    """Load already-preprocessed .npz windows without signal preprocessing."""

    def __init__(self, records: Sequence[SampleRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]

        with np.load(record.path, allow_pickle=False) as data:
            x = data["x"]

        if x.shape != (EXPECTED_INPUT_CHANNELS, EXPECTED_WINDOW_SAMPLES):
            raise ValueError(
                f"Unexpected shape {x.shape} in {record.path}; expected "
                f"({EXPECTED_INPUT_CHANNELS}, {EXPECTED_WINDOW_SAMPLES})."
            )

        # Copy to a normal contiguous float32 array. No filtering, downsampling,
        # normalization, resampling, or window extraction is applied here.
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
        y_tensor = torch.tensor(CLASS_TO_INDEX[record.gesture], dtype=torch.long)

        return x_tensor, y_tensor
