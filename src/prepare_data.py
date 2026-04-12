import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
VAL_SIZE = 0.2
SEED     = 42
IMG_EXTS = [".jpg", ".jpeg", ".bmp", ".png"]


def prepare():
    classes = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    if not classes:
        print("No dataset found in data/raw/")
        return

    print(f"Found {len(classes)} classes: {classes}\n")

    for split in ["train", "val"]:
        for cls in classes:
            (PROC_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0

    for cls in classes:
        images = []
        for ext in IMG_EXTS:
            images += list((RAW_DIR / cls).glob(f"*{ext}"))
            images += list((RAW_DIR / cls).glob(f"*{ext.upper()}"))

        if not images:
            print(f"  {cls}: no images found")
            continue

        train_imgs, val_imgs = train_test_split(images, test_size=VAL_SIZE, random_state=SEED)

        for img in train_imgs:
            shutil.copy(img, PROC_DIR / "train" / cls / img.name)
        for img in val_imgs:
            shutil.copy(img, PROC_DIR / "val" / cls / img.name)

        print(f"  {cls:15s} train: {len(train_imgs):4d} | val: {len(val_imgs):4d}")
        total_train += len(train_imgs)
        total_val   += len(val_imgs)

    print(f"\nDone. train: {total_train} | val: {total_val}")


if __name__ == "__main__":
    prepare()
