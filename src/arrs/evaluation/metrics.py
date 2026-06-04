import json
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    ndcg_score,
    average_precision_score
)

from sklearn.calibration import calibration_curve


ARTIFACT_DIR = Path("artifacts/evaluation")
ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def precision_at_k(y_true, y_scores, k=10):

    ranked = sorted(
        zip(y_scores, y_true),
        reverse=True
    )[:k]

    relevant = sum(label for _, label in ranked)

    return relevant / k


def evaluate_model(
    y_true,
    y_scores
):

    metrics = {
        "auc": roc_auc_score(
            y_true,
            y_scores
        ),
        "log_loss": log_loss(
            y_true,
            y_scores
        ),
        "precision_at_10": precision_at_k(
            y_true,
            y_scores,
            k=10
        ),
        "ndcg": ndcg_score(
            [y_true],
            [y_scores]
        ),
        "map": average_precision_score(
            y_true,
            y_scores
        )
    }

    return metrics


def save_metrics(metrics):

    output_file = (
        ARTIFACT_DIR /
        "metrics.json"
    )

    with open(
        output_file,
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"Saved metrics -> {output_file}"
    )


def save_calibration_plot(
    y_true,
    y_scores
):

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_scores,
        n_bins=10
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        prob_pred,
        prob_true,
        marker="o"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Probability"
    )

    plt.ylabel(
        "Observed Frequency"
    )

    plt.title(
        "Calibration Curve"
    )

    output_file = (
        ARTIFACT_DIR /
        "calibration_curve.png"
    )

    plt.savefig(
        output_file,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved calibration plot -> {output_file}"
    )