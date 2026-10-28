# Attention Mechanisms for Surface Vibration-based Gesture Recognition

## Least Parameters Configuration

This repository contains the code for evaluating four lightweight attention mechanisms on the **Least Parameters** backbone (1,802 parameters, 4 blocks, 16 channels, kernel size 9) from the surface-vibration gesture recognition system of [Hettstedt et al. (2026)](https://arxiv.org/abs/2605.10110).

This work is part of a Bachelor thesis at the **Universität Duisburg-Essen**, Intelligent Embedded Systems Group.

> **Companion repository:** The [Best Accuracy configuration](https://github.com/Jonathanloic/best-accuracy-gesture-recognition-r-sweep) (8,722 parameters, 6 blocks, 32 channels, kernel size 15) is evaluated in a separate repository with the same structure.

---

## Attention Variants

| Variant | Mechanism | Parameters Added | Hyperparameter |
|---|---|---|---|
| Baseline | None | 0 | — |
| SE | Channel (bottleneck) | 128–1,024 | r ∈ {2, 4, 8, 16} |
| ECA | Channel (local conv) | 12 | k = 3 (derived) |
| Temporal | Temporal (conv over time) | 56 | k = 7 |
| Channel–Temporal | Channel then temporal | 184–1,080 | r ∈ {2, 4, 8, 16}, k = 7 |

All modules are inserted after ReLU and before Dropout in every depthwise-separable block.

---

## Repository Structure

```
├── model.py                          # Baseline depthwise-separable 1D-CNN
├── model_se.py                       # Squeeze-and-Excitation module
├── model_eca.py                      # Efficient Channel Attention module
├── model_temporal.py                 # Temporal attention module
├── model_attention.py                # Channel–temporal attention module
├── dataset.py                        # Data loading and preprocessing
├── train.py                          # Training loop
├── experiments.py                    # Single LOSO evaluation run
├── repeat_experiments.py             # 20-run orchestration script
├── reproducibility.py                # Seeding and deterministic configuration
├── sweep_attention_reduction_ratio.py # Reduction ratio sweep launcher
├── verify_temporal_model.py          # Temporal model verification
└── results/                          # Experimental results (JSON summaries)
```

---

## Evaluation Protocol

- **Cross-validation:** Leave-one-subject-out (LOSO), 15 folds
- **Repetitions:** 20 independent runs (seeds 42–61)
- **Epochs:** 300 (fixed, no early stopping)
- **Optimiser:** AdamW, lr = 1×10⁻³, weight decay = 0.01 (default)
- **Batch size:** 128
- **Loss:** Cross-entropy, unweighted
- **Determinism:** cuDNN deterministic enabled, autotuning disabled

Every reported result is the mean ± population standard deviation over the 20 runs.

---

## Usage

### Single Evaluation Run

```bash
python experiments.py \
    --model least_parameters_se \
    --protocol loso \
    --data-root /path/to/preprocessed-data \
    --output-dir results/se_r2 \
    --seed 42 \
    --epochs 300 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --se-reduction-ratio 2
```

### Full 20-Run Evaluation

```bash
python repeat_experiments.py \
    --model least_parameters_se \
    --protocol loso \
    --data-root /path/to/preprocessed-data \
    --output-dir results/se_r2 \
    --runs 20 \
    --base-seed 42 \
    --se-reduction-ratio 2
```

### Reduction Ratio Sweep

```bash
python sweep_attention_reduction_ratio.py \
    --model least_parameters_se \
    --data-root /path/to/preprocessed-data \
    --output-dir results/
```

This runs the full 20-run evaluation for r ∈ {2, 4, 8, 16} sequentially.

---

## Key Results

| Variant | Clean (%) | Noisy (%) | Parameters |
|---|---|---|---|
| Baseline | 83.13 ± 2.68 | 54.29 ± 2.23 | 1,802 |
| SE (r=2) | 86.68 ± 1.21 | 59.54 ± 0.80 | 2,826 |
| ECA | 85.93 ± 1.34 | 58.28 ± 0.92 | 1,814 |
| Temporal | 85.75 ± 2.03 | 53.28 ± 2.07 | 1,858 |
| **CT (r=2)** | **87.24 ± 0.77** | **60.10 ± 0.63** | **2,882** |

Channel–temporal attention at r=2 achieves the best accuracy. ECA achieves 233–333 pp/1k parameter efficiency — two orders of magnitude above any other mechanism.

---

## Requirements

- Python 3.10+
- PyTorch (with CUDA support)
- NumPy

---

## Dataset

The preprocessed surface-vibration gesture corpus is from [Hettstedt et al. (2026)](https://arxiv.org/abs/2605.10110). It contains 8,999 gesture events from 15 participants across 6 classes (swipe-left, swipe-right, swipe-up, swipe-down, tap, knock), recorded with 4 piezoelectric sensors at 1 kHz, downsampled by factor 2 to 500 Hz effective rate.

**The dataset is not included in this repository.** Contact the [Intelligent Embedded Systems Group](https://www.2is.uni-due.de/) at the Universität Duisburg-Essen for access.

---

## Citation

If you use this code, please cite:

```bibtex
@thesis{fondjo2026attention,
  author  = {Fondjo Kamtchueng, Jonathan Loic},
  title   = {Attention Mechanisms for Surface Vibration-based
             Gesture Recognition},
  school  = {Universität Duisburg-Essen},
  year    = {2026},
  type    = {Bachelor Thesis}
}
```

---

## License

This project was developed as part of an academic thesis. Please contact the author for usage terms.
