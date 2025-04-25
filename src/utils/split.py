import random
import shutil
from pathlib import Path

def split_clean_images(source_dir, temp_70_dir=None, temp_30_dir=None, ratio=0.7, seed=42):
    source_dir = Path(source_dir)

    all_images = sorted([
        p for p in source_dir.rglob("*.tif")
        if "負極" in p.name and "Z軸" in str(p.parent)
        and 215 <= int(p.stem.split("_")[-1]) <= 222
    ])

    random.seed(seed)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * ratio)
    images_70 = all_images[:split_idx]
    images_30 = all_images[split_idx:]

    if temp_70_dir:
        temp_70_dir = Path(temp_70_dir)
        for img in images_70:
            rel_path = img.relative_to(source_dir)
            dst = temp_70_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)

    if temp_30_dir:
        temp_30_dir = Path(temp_30_dir)
        for img in images_30:
            rel_path = img.relative_to(source_dir)
            dst = temp_30_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)

    print(f"✅ Split {len(all_images)} images → {len(images_70)} in 70%, {len(images_30)} in 30%")
    return str(temp_70_dir) if temp_70_dir else None, str(temp_30_dir) if temp_30_dir else None
