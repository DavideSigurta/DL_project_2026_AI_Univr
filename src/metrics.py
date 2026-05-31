from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.manifold import TSNE


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba)
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain at least two classes to compute AUC.")

    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision_malignant": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_malignant": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    metrics["confusion_matrix"] = cm
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_roc_curve(y_true, y_pred_proba, label: str = "", ax=None, save_path: Optional[str] = None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure
    ax.plot(fpr, tpr, label=label or "ROC")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    if label:
        ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_auc_vs_labels(results_dict: Dict[str, Dict[float, float]], save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, values in results_dict.items():
        xs = sorted(values.keys())
        ys = [values[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Label fraction")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs Label Budget")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_tsne(features, labels, domain_labels=None, save_path: Optional[str] = None):
    features = np.asarray(features)
    labels = np.asarray(labels).astype(int)
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
    emb = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(5, 4))
    if domain_labels is None:
        ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="coolwarm", s=10, alpha=0.7)
    else:
        domain_labels = np.asarray(domain_labels).astype(int)
        colors = ["tab:blue", "tab:orange"]
        markers = ["o", "s"]
        for d in [0, 1]:
            mask_d = domain_labels == d
            for c in [0, 1]:
                mask = mask_d & (labels == c)
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    c=colors[d],
                    marker=markers[c],
                    s=12,
                    alpha=0.7,
                    label=f"domain {d} / class {c}",
                )
        ax.legend(fontsize=8, ncol=2, frameon=False)

    ax.set_title("t-SNE Features")
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
