from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.datasets import build_loaders
from src.models import build_backbone
from src.utils import ensure_dir, get_device

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_latest_run(exp_name: str, base_dir: str = "results/runs") -> Optional[Path]:
    """Return the most recent run directory for a given experiment name."""
    base = Path(base_dir) / exp_name
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    return runs[-1] if runs else None

def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all ground truth labels and predicted probabilities."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).view(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred_proba = np.concatenate(all_probs)
    return y_true, y_pred_proba


def _get_eval_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_path = run_dir / "checkpoints/best.pt"
    if ckpt_path.exists():
        return ckpt_path
    ckpt_path = run_dir / "checkpoints/last.pt"
    if ckpt_path.exists():
        return ckpt_path
    return None


def evaluate_on_csv(
    run_dir: Path,
    base_config: dict,
    test_csv: str,
    train_csv: Optional[str] = None,
    batch_size: Optional[int] = None,
    device=None,
    metrics_name: str = "test_metrics.json",
) -> Optional[dict]:
    """Evaluate checkpoint on a given CSV and cache metrics in run_dir."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / metrics_name
    if metrics_path.exists():
        with open(metrics_path) as fp:
            return json.load(fp)

    ckpt_path = _get_eval_checkpoint(run_dir)
    if ckpt_path is None:
        return None

    cfg = copy.deepcopy(base_config)
    cfg.setdefault("data", {})
    cfg["data"]["task"] = "supervised"
    cfg["data"]["test_csv"] = test_csv
    if batch_size is not None:
        cfg["data"]["batch_size"] = batch_size
    if train_csv is not None:
        cfg["data"]["train_csv"] = train_csv
    else:
        cfg["data"].setdefault("train_csv", "data/processed/isic2018/train.csv")

    if device is None:
        device = get_device(cfg.get("experiment", {}).get("device"))

    loader = build_loaders(cfg)["test"]
    model = build_backbone(
        arch=cfg["model"]["arch"],
        pretrained=False,
        num_classes=cfg["model"].get("num_classes", 1),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    y_true, y_pred_proba = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred_proba)
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp, indent=2, cls=NumpyEncoder)
    return metrics

def evaluate_and_report(run_dir: Path, model, loader, device) -> dict:
    """Evaluate the best checkpoint, save metrics and figures, print summary."""
    # Load best checkpoint
    ckpt_path = run_dir / "checkpoints/best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints/last.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found in {run_dir / 'checkpoints'}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # Collect predictions and compute metrics
    y_true, y_pred_proba = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred_proba)

    # Save test metrics
    metrics_path = run_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    # Generate figures
    figures_dir = ensure_dir(run_dir / "figures")
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=figures_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_pred_proba, label=f"AUC = {metrics['auc_roc']:.3f}", save_path=figures_dir / "roc_curve.png")

    # Print summary
    print("\n=== Test Set Evaluation ===")
    print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"  Precision (Malig):  {metrics['precision_malignant']:.4f}")
    print(f"  Recall (Malig):     {metrics['recall_malignant']:.4f}")
    print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")
    print(f"\nArtifacts saved to: {run_dir}")
    return metrics

def plot_training_history(run_dir: Path, save_path: Optional[str] = None):
    """Plot training/validation loss and validation AUC from metrics.jsonl."""
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print("No metrics.jsonl found. Cannot plot training history.")
        return

    records = []
    with open(metrics_file, "r") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Loss plot
    if "train_loss" in df and "val_loss" in df:
        axes[0].plot(df["epoch"], df["train_loss"], label="Train loss")
        axes[0].plot(df["epoch"], df["val_loss"], label="Val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Loss curves")
    # AUC plot
    if "val_auc" in df:
        axes[1].plot(df["epoch"], df["val_auc"], label="Val AUC", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].legend()
        axes[1].set_title("Validation AUC")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def collect_auc_vs_fraction(exp_name_prefix: str, fractions: list, base_dir="results/runs"):
    """Collect AUC from test_metrics.json for multiple runs.

    Args:
        exp_name_prefix: e.g., "baseline_resnet18_{:.2f}" (will format with fraction)
        fractions: list of floats, e.g., [0.01, 0.05, 0.10]
        base_dir: root runs directory

    Returns:
        dict: fraction -> AUC (or None if missing)
    """
    aucs = {}
    for f in fractions:
        exp_name = exp_name_prefix.format(f)
        run_dir = get_latest_run(exp_name, base_dir)
        if run_dir is None:
            print(f"Warning: no run found for {exp_name}")
            aucs[f] = None
            continue
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as fp:
                data = json.load(fp)
                aucs[f] = data.get("auc_roc")
        else:
            print(f"Warning: test_metrics.json missing for {exp_name}")
            aucs[f] = None
    return aucs