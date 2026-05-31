"""
data/scripts/inject_noise.py
Inject symmetric label noise into ISIC 2018 train.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NOISE_LEVELS = [0.10, 0.20, 0.30]

TRAIN_CSV = Path("data/processed/isic2018/train.csv")
OUT_DIR = Path("data/processed/isic2018/noisy")


def main() -> None:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing {TRAIN_CSV}. Run data preparation first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in train CSV.")

    rng = np.random.default_rng(SEED)
    n = len(df)

    for level in NOISE_LEVELS:
        k = int(round(n * level))
        idx = rng.choice(n, size=k, replace=False)
        noisy = df.copy()
        noisy.loc[idx, "target"] = 1 - noisy.loc[idx, "target"]

        pct = int(round(level * 100))
        out_csv = OUT_DIR / f"train_noise{pct:02d}.csv"
        noisy.to_csv(out_csv, index=False)

        mask = pd.DataFrame(
            {
                "image_id": df["image_id"],
                "is_noisy": df.index.isin(idx).astype(int),
            }
        )
        out_mask = OUT_DIR / f"noise_mask{pct:02d}.csv"
        mask.to_csv(out_mask, index=False)
        print(f"[OK] {out_csv} (+ {out_mask})")


if __name__ == "__main__":
    main()
