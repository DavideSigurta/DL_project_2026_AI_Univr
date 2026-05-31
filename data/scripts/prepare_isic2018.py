"""
data/scripts/prepare_isic2018.py
Downloads HAM10000 from Kaggle if not present, then processes:
binary labels (MEL vs rest) and a stratified 80/10/10 split.

Dataset: surajghuwalewala/ham1000-segmentation-and-classification
Expected structure after unzip:
  data/raw/isic2018/
  ├── images/
  │   ├── ISIC_0024306.jpg
  │   └── ...
  ├── masks/          (unused)
  └── GroundTruth.csv  (columns: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC)
"""
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DIR  = Path("data/raw/isic2018")
OUT_DIR  = Path("data/processed/isic2018")
IMG_DIR  = RAW_DIR / "images"
CSV_PATH = RAW_DIR / "GroundTruth.csv"

KAGGLE_DATASET = "surajghuwalewala/ham1000-segmentation-and-classification"
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MIN_IMAGES = 10000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def download_if_needed():
    imgs_ok = IMG_DIR.exists() and len(list(IMG_DIR.glob("*.jpg"))) >= MIN_IMAGES
    csv_ok  = CSV_PATH.exists()
    if imgs_ok and csv_ok:
        n = len(list(IMG_DIR.glob("*.jpg")))
        print(f"[INFO] ISIC 2018 already present ({n} images). Skipping download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Downloading HAM10000 from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(RAW_DIR), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[STDERR]", result.stderr)
        raise RuntimeError(
            "Download failed.\n"
            "Check: 1) kaggle installed (`pip install kaggle`)\n"
            "       2) token at ~/.kaggle/access_token or ~/.kaggle/kaggle.json\n"
            "       3) dataset terms accepted on Kaggle"
        )
    print("[INFO] Download completed.")


def find_images_dir():
    """Find the images folder, handling possible layout variants."""
    candidates = [
        RAW_DIR / "images",
        RAW_DIR / "HAM10000_images",
        RAW_DIR,
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("*.jpg"))) > 1000:
            return c
    raise FileNotFoundError(
        f"Images folder not found under {RAW_DIR}. "
        f"Expected: {RAW_DIR}/images/*.jpg"
    )


def find_csv():
    """Find the ground-truth CSV, handling name variants."""
    candidates = [
        RAW_DIR / "GroundTruth.csv",
        RAW_DIR / "groundtruth.csv",
        RAW_DIR / "ISIC2018_Task3_Training_GroundTruth.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Ground-truth CSV not found under {RAW_DIR}. "
        f"Expected: GroundTruth.csv"
    )


def process():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_dir = find_images_dir()
    csv_path = find_csv()

    print(f"[INFO] Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalize column names (case-insensitive).
    df.columns = [c.upper() for c in df.columns]

    # Validate required columns.
    for col in ["IMAGE"] + CLASSES:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

    # Binary labels: 1 = malignant (MEL), 0 = benign (all others).
    df["target"] = (df["MEL"] > 0.5).astype(int)
    df["image_id"] = df["IMAGE"].astype(str)
    df["source"] = "isic2018"

    # Filepath.
    df["filepath"] = df["image_id"].apply(lambda x: str(img_dir / f"{x}.jpg"))

    # Spot-check that files exist.
    sample_missing = [p for p in df["filepath"].sample(min(10, len(df))) if not Path(p).exists()]
    if sample_missing:
        raise FileNotFoundError(
            f"Some image files were not found: {sample_missing[:3]}\n"
            f"Check image directory: {img_dir}"
        )

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Stratified 80/10/10 split.
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=df["target"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        random_state=42,
        stratify=temp_df["target"],
    )

    cols = ["image_id", "filepath", "target", "source"]
    train_df[cols].reset_index(drop=True).to_csv(OUT_DIR / "train.csv", index=False)
    val_df[cols].reset_index(drop=True).to_csv(OUT_DIR / "val.csv", index=False)
    test_df[cols].reset_index(drop=True).to_csv(OUT_DIR / "test.csv", index=False)

    n_total = len(df)
    n_mal_tr = train_df["target"].sum()
    n_mal_val = val_df["target"].sum()
    n_mal_test = test_df["target"].sum()
    print(f"[INFO] Total samples: {n_total}")
    print(f"[INFO] Train: {len(train_df)} ({n_mal_tr} malignant, {n_mal_tr/len(train_df):.1%})")
    print(f"[INFO] Val:   {len(val_df)} ({n_mal_val} malignant, {n_mal_val/len(val_df):.1%})")
    print(f"[INFO] Test:  {len(test_df)} ({n_mal_test} malignant, {n_mal_test/len(test_df):.1%})")
    print(f"[OK] CSV files saved to {OUT_DIR}")


if __name__ == "__main__":
    download_if_needed()
    process()
