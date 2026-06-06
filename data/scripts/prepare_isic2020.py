import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DIR  = Path("data/raw/isic2020")
OUT_DIR  = Path("data/processed/isic2020")

KAGGLE_DATASET = "nischaydnk/isic-2020-jpg-224x224-resized"
MIN_IMAGES = 33000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Official ISIC 2020 duplicate list (425 images)
# https://challenge2020.isic-archive.com  → "Download duplicate image list"
DUPLICATES_PATH = RAW_DIR / "ISIC_2020_Training_Duplicates.csv"


def download_if_needed():
    # Recursive check (handles both train-image/ and train-image/image/).
    existing = list(RAW_DIR.rglob("*.jpg"))
    if len(existing) >= MIN_IMAGES:
        print(f"[INFO] ISIC 2020 already present ({len(existing)} images). Skipping download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] Downloading ISIC 2020 (224x224) from Kaggle...")
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
    """Find the folder that actually contains the .jpg files."""
    candidates = [
        RAW_DIR / "train-image",
        RAW_DIR / "train-image" / "image",
        RAW_DIR / "train",
        RAW_DIR / "train" / "image",
        RAW_DIR,
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("*.jpg"))) > 1000:
            return c

    # Fallback: pick the folder with the most jpg files.
    all_dirs = {p.parent for p in RAW_DIR.rglob("*.jpg")}
    if all_dirs:
        best = max(all_dirs, key=lambda d: len(list(d.glob("*.jpg"))))
        if len(list(best.glob("*.jpg"))) > 1000:
            print(f"[WARN] Images folder found in a non-standard location: {best}")
            return best

    raise FileNotFoundError(
        f"No folder with .jpg images found under {RAW_DIR}.\n"
        f"Expected: {RAW_DIR}/train-image/*.jpg"
    )


def find_csv():
    """Find the metadata CSV, handling name variants."""
    candidates = [
        RAW_DIR / "train-metadata.csv",
        RAW_DIR / "train.csv",
        RAW_DIR / "ISIC_2020_Training_GroundTruth.csv",
        RAW_DIR / "train_metadata.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Metadata CSV not found under {RAW_DIR}.\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


def resolve_id_column(df):
    """Return the image ID column name (handles isic_id / image_name)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in ["isic_id", "image_name", "image", "id"]:
        if candidate in cols_lower:
            return cols_lower[candidate]
    raise ValueError(
        f"No image ID column found.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: isic_id, image_name, image"
    )


def resolve_target_column(df):
    """Return the binary target column (target or benign_malignant)."""
    cols_lower = {c.lower(): c for c in df.columns}
    if "target" in cols_lower:
        return cols_lower["target"], False  # (nome_colonna, needs_mapping)
    if "benign_malignant" in cols_lower:
        return cols_lower["benign_malignant"], True
    raise ValueError(
        f"No target column found.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected: 'target' or 'benign_malignant'"
    )


def load_duplicates():
    """Load the official list of 425 duplicates if available."""
    if not DUPLICATES_PATH.exists():
        return set()
    try:
        dup_df = pd.read_csv(DUPLICATES_PATH)
        # The official CSV has columns like 'image_name_1', 'image_name_2'.
        dup_ids = set()
        for col in dup_df.columns:
            if "image" in col.lower() or "isic" in col.lower():
                dup_ids.update(dup_df[col].dropna().astype(str).tolist())
        print(f"[INFO] Found {len(dup_ids)} duplicate IDs to remove.")
        return dup_ids
    except Exception as e:
        print(f"[WARN] Could not load duplicates list: {e}")
        return set()


def process():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_dir  = find_images_dir()
    csv_path = find_csv()

    print(f"[INFO] Images folder: {img_dir}")
    print(f"[INFO] Metadata CSV:  {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Resolve columns.
    id_col = resolve_id_column(df)
    tgt_col, needs_mapping = resolve_target_column(df)
    print(f"[INFO] ID column: '{id_col}' | Target column: '{tgt_col}'")

    # Map benign_malignant → 0/1 if needed.
    if needs_mapping:
        mapping = {"benign": 0, "malignant": 1}
        df["target"] = df[tgt_col].str.lower().map(mapping)
        if df["target"].isna().any():
            unique_vals = df[tgt_col].unique()
            raise ValueError(
                f"Unexpected values in '{tgt_col}': {unique_vals}. "
                f"Expected: 'benign' / 'malignant'"
            )
        tgt_col = "target"

    # Normalize ID column to 'isic_id'.
    if id_col != "isic_id":
        df = df.rename(columns={id_col: "isic_id"})
        print(f"[INFO] Renamed '{id_col}' to 'isic_id'")

    # Remove official duplicates.
    dup_ids = load_duplicates()
    if dup_ids:
        before = len(df)
        df = df[~df["isic_id"].astype(str).isin(dup_ids)].reset_index(drop=True)
        print(f"[INFO] Removed {before - len(df)} duplicates ({before} → {len(df)})")

    # Filepath.
    df["filepath"] = df["isic_id"].apply(lambda x: str(img_dir / f"{x}.jpg"))
    df["image_id"] = df["isic_id"].astype(str)
    df["source"] = "isic2020"

    # Spot-check that files exist.
    sample_missing = [p for p in df["filepath"].sample(min(10, len(df)), random_state=0)
                      if not Path(p).exists()]
    if sample_missing:
        raise FileNotFoundError(
            f"Some image files were not found: {sample_missing[:3]}\n"
            f"Check images folder: {img_dir}\n"
            f"Example expected path: {df['filepath'].iloc[0]}"
        )

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Stratified 80/10/10 split.
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
        stratify=df[tgt_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        random_state=42,
        stratify=temp_df[tgt_col],
    )

    cols = ["image_id", "filepath", tgt_col, "source"]
    if tgt_col != "target":
        train_df = train_df.rename(columns={tgt_col: "target"})
        val_df = val_df.rename(columns={tgt_col: "target"})
        cols = ["image_id", "filepath", "target", "source"]

    train_df[cols].reset_index(drop=True).to_csv(OUT_DIR / "train.csv", index=False)
    val_df[cols].reset_index(drop=True).to_csv(OUT_DIR / "val.csv", index=False)
    test_df[cols].reset_index(drop=True).to_csv(OUT_DIR / "test.csv", index=False)

    n_mal_tr = train_df["target"].sum()
    n_mal_val = val_df["target"].sum()
    n_mal_test = test_df["target"].sum()
    print(f"[INFO] Train: {len(train_df)} ({n_mal_tr} malignant, {n_mal_tr/len(train_df):.2%})")
    print(f"[INFO] Val:   {len(val_df)} ({n_mal_val} malignant, {n_mal_val/len(val_df):.2%})")
    print(f"[INFO] Test:  {len(test_df)} ({n_mal_test} malignant, {n_mal_test/len(test_df):.2%})")
    print(f"[OK]  CSV files saved to {OUT_DIR}")


if __name__ == "__main__":
    download_if_needed()
    process()
