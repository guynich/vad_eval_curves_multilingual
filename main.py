"""Evaluate VAD model performance using AUC curves with Hugging Face dataset."""

from dataclasses import dataclass
from typing import List
import pprint

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from speech_detector import SpeechDetector

@dataclass
class BinaryMetrics:
    """Metrics for binary classification evaluation."""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float

@dataclass
class AUCMetrics:
    """Area Under Curve metrics."""
    roc_auc: float
    pr_auc: float
    count: int
    count_speech_true: int
    active_speech_hours: float


def compute_overall_auc(y_bools: List[List[int]], y_scores: List[List[float]]) -> AUCMetrics:
    """Compute ROC and PR AUC scores for flattened predictions."""
    flat_bools = np.concatenate(y_bools)
    flat_scores = np.concatenate(y_scores)

    return AUCMetrics(
        roc_auc=roc_auc_score(flat_bools, flat_scores),
        pr_auc=average_precision_score(flat_bools, flat_scores),
        count=len(flat_bools),
        count_speech_true=sum(flat_bools),
        active_speech_hours=sum(flat_bools) * 512 / (16000 * 3600),
    )


def evaluate_binary_classification(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    """Evaluate binary classification predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return BinaryMetrics(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0)
    )


def plot_performance_curves(
    y_true: List[List[int]],
    y_scores: List[List[float]],
    split: str = "",
    confidence_label: str = "",
    threshold_markers: List[float] = [0.3, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0]
) -> None:
    """Plot ROC and PR curves with threshold markers."""
    flat_true = np.concatenate(y_true)
    flat_scores = np.concatenate(y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(flat_true, flat_scores)
    roc_auc = roc_auc_score(flat_true, flat_scores)

    ax1.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", label="Random")
    _add_threshold_markers(ax1, fpr, tpr, roc_thresholds, threshold_markers)

    ax1.set(xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"{split}: ROC Curve{confidence_label}")
    ax1.grid()
    ax1.legend()

    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(flat_true, flat_scores)
    pr_auc = average_precision_score(flat_true, flat_scores)

    ax2.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}", lw=2)
    _add_threshold_markers(ax2, recall, precision, pr_thresholds, threshold_markers)

    ax2.set(xlabel="Recall",
            ylabel="Precision",
            title=f"{split}: Precision-Recall Curve{confidence_label}",
            ylim=[0.75, 1.02])
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show(block=False)


def _add_threshold_markers(ax: plt.Axes,
                         x_coords: np.ndarray,
                         y_coords: np.ndarray,
                         thresholds: np.ndarray,
                         target_thresholds: List[float]) -> None:
    """Add threshold markers to plot."""
    for target in target_thresholds:
        idx = np.abs(thresholds - target).argmin()
        ax.plot(x_coords[idx], y_coords[idx], "ro")
        ax.annotate(f"{thresholds[idx]:.2f}",
                   (x_coords[idx], y_coords[idx]),
                   fontsize=8,
                   ha="right")


def process_audio(audio: np.ndarray, detector: SpeechDetector) -> List[float]:
    """Process audio chunks and return VAD probabilities."""
    detector.reset()
    chunk_size = detector.chunk_size
    return [
        detector(audio[i:i + chunk_size])
        for i in range(0, (len(audio) // chunk_size) * chunk_size, chunk_size)
    ]


def main():
    SPLITS = [
        "test.dutch",
        "test.french",
        "test.german",
        "test.italian",
        "test.polish",
        "test.portuguese",
        "test.spanish",
    ]
    VAD_THRESHOLD = 0.5

    dataset = load_dataset("guynich/multilingual_librispeech_test_vad")
    detector = SpeechDetector()
    results = {}

    for split in SPLITS:
        all_speech, all_vad_probs = [], []
        confident_speech, confident_vad_probs = [], []

        for idx, example in enumerate(dataset[split]):
            speech = example["speech"]
            confidence = np.array(example["confidence"])
            vad_probs = process_audio(example["audio"]["array"], detector)

            # Store all results and confidence-filtered results
            all_speech.append(speech)
            all_vad_probs.append(vad_probs)

            confident_mask = confidence == 1
            confident_speech.append(np.array(speech)[confident_mask])
            confident_vad_probs.append(np.array(vad_probs)[confident_mask])

            # Evaluate predictions for fixed threshold
            metrics = evaluate_binary_classification(
                np.array(speech),
                np.array(vad_probs) > VAD_THRESHOLD
            )

            confident_metrics = evaluate_binary_classification(
                confident_speech[-1],
                confident_vad_probs[-1] > VAD_THRESHOLD
            )

            print(f"\nExample: [{idx:04d}]  Split: {split}")
            print("All data metrics:")
            pprint.pprint(metrics)
            print("Data metrics excluding low confidence features:")
            pprint.pprint(confident_metrics)

        # Compute and plot overall results
        results[split] = compute_overall_auc(all_speech, all_vad_probs)
        results[f"{split}_confidence"] = compute_overall_auc(
            confident_speech, confident_vad_probs
        )

        plot_performance_curves(all_speech, all_vad_probs, split=split)
        plot_performance_curves(
            confident_speech,
            confident_vad_probs,
            split=split,
            confidence_label=" (exclude low confidence)"
        )

    print("\nOverall results:")
    pprint.pprint(results)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
