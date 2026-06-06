from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def make_combined_csv(
    isic2018_csv: str = "data/processed/isic2018/train.csv",
    isic2020_csv: str = "data/processed/isic2020/train.csv",
    output_csv: str = "data/processed/combined_unlabeled.csv",
) -> str:
    df18 = pd.read_csv(isic2018_csv)
    df20 = pd.read_csv(isic2020_csv)

    for name, df in [("isic2018", df18), ("isic2020", df20)]:
        missing = [c for c in ["image_id", "filepath", "target", "source"] if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

    combined = pd.concat([df18, df20], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print(f"isic2018 train : {len(df18):>6} rows")
    print(f"isic2020 train : {len(df20):>6} rows")
    print(f"combined       : {len(combined):>6} rows  →  {output_csv}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--isic2018", default="data/processed/isic2018/train.csv")
    parser.add_argument("--isic2020", default="data/processed/isic2020/train.csv")
    parser.add_argument("--output", default="data/processed/combined_unlabeled.csv")
    args = parser.parse_args()
    make_combined_csv(args.isic2018, args.isic2020, args.output)