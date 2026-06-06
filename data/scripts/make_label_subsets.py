from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50]

TRAIN_CSV = Path("data/processed/isic2018/train.csv")
OUT_DIR = Path("data/processed/isic2018/subsets")


def _subset_filename(frac: float) -> str:
    pct = int(round(frac * 100))
    return f"train_{pct:02d}pct.csv"


def main() -> None:
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing {TRAIN_CSV}. Run data preparation first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in train CSV.")

    for frac in FRACTIONS:
        subset, _ = train_test_split(
            df,
            train_size=frac,
            random_state=SEED,
            stratify=df["target"],
        )
        subset = subset.reset_index(drop=True)
        out_path = OUT_DIR / _subset_filename(frac)
        subset.to_csv(out_path, index=False)
        print(f"[OK] {out_path} ({len(subset)} samples)")


if __name__ == "__main__":
    main()
