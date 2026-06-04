from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> torch.device:
    preferred = (preferred or "").lower().strip()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def init_run_dir(exp_name: str, base_dir: str | Path = "results/runs") -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = exp_name.replace(" ", "_")
    run_dir = Path(base_dir) / safe_name / timestamp
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "figures")
    return run_dir


def setup_logging(run_dir: str | Path, name: str = "run") -> logging.Logger:
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: Optional[str | torch.device] = None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta
        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def cleanup() -> None:
    """Release memory between experiment runs.

    Call after each training loop iteration to prevent OOM
    (accumulated models, tensors, DataLoader workers, matplotlib figures).
    Safe to call anywhere — uses lazy imports and graceful fallbacks.
    """
    gc.collect()

    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Clear Apple MPS (Metal) cache — graceful fallback if API missing
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    # Close all matplotlib figures
    try:
        import matplotlib.pyplot as plt  # lazy import
        plt.close('all')
    except Exception:
        pass
