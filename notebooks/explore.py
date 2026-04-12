import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import Counter

RAW_DIR = Path("data/processed/train")


def explore():
    print("=" * 50)
    print("Dataset Explorer")
    print("=" * 50)

    classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    if not classes:
        print("No dataset found in data/raw/")
        return

    IMG_EXTS = [".jpg", ".jpeg", ".bmp", ".png"]
    class_counts = {}
    sample_images = {}

    for cls in classes:
        images = []
        for ext in IMG_EXTS:
            images += list((RAW_DIR / cls).glob(f"*{ext}"))
        class_counts[cls] = len(images)
        if images:
            sample_images[cls] = images[0]

    print(f"\nFound {len(classes)} classes:\n")
    for cls, count in class_counts.items():
        bar = "█" * (count // 10)
        print(f"  {cls:20s}: {count:4d}  {bar}")

    total = sum(class_counts.values())
    print(f"\n  {'TOTAL':20s}: {total:4d}")

    if sample_images:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        for i, (cls, img_path) in enumerate(sample_images.items()):
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(f"{cls}\n({class_counts[cls]} images)", fontsize=10)
            axes[i].axis("off")

        for j in range(len(sample_images), len(axes)):
            axes[j].axis("off")

        plt.suptitle("NEU Steel Surface Defect — Sample Images", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig("notebooks/dataset_overview.png", bbox_inches='tight', dpi=150)
        print("Saved → notebooks/dataset_overview.png")
        plt.show()

    print("\nImage sizes:")
    sizes = []
    for cls in classes:
        for ext in IMG_EXTS:
            for img_path in list((RAW_DIR / cls).glob(f"*{ext}"))[:5]:
                img = Image.open(img_path)
                sizes.append(img.size)

    for size, count in Counter(sizes).most_common(5):
        print(f"  {size[0]}×{size[1]} px → {count}")

    print("\nDone. Next: python src/prepare_data.py")


if __name__ == "__main__":
    explore()
