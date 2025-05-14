import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import json


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

    print(f"Split {len(all_images)} images → {len(images_70)} in 70%, {len(images_30)} in 30%")
    return str(temp_70_dir) if temp_70_dir else None, str(temp_30_dir) if temp_30_dir else None


def split_and_save_coco_dataset(coco_images, coco_annotations, output_base_dir, original_image_dir, split_ratios=(0.7, 0.2, 0.1)):
    print(f"Splitting dataset into {split_ratios[0]*100:.0f}% train, {split_ratios[1]*100:.0f}% val, {split_ratios[2]*100:.0f}% test")
    if len(coco_images) > 10000:
        coco_images = random.sample(coco_images, 10000)

    img_train_val, img_test = train_test_split(coco_images, test_size=split_ratios[2], random_state=42)
    img_train, img_val = train_test_split(
        img_train_val,
        test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]),
        random_state=42
    )
    img_train += [img for img in coco_images if "_crop_original" in img["file_name"] and img not in img_train and img not in img_val and img not in img_test]

    split_data = {"train": img_train, "valid": img_val, "test": img_test}

    def filter_annotations(image_subset, all_annotations):
        image_ids = {img["id"] for img in image_subset}
        return [ann for ann in all_annotations if ann["image_id"] in image_ids]

    for split, imgs in split_data.items():
        split_dir = os.path.join(output_base_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for img in imgs:
            src = os.path.join(original_image_dir, img["file_name"])
            dst = os.path.join(split_dir, img["file_name"])
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                print(f"Missing file skipped: {src}")

        split_ann = {
            "info": {
                "year": 2025,
                "version": "1.0",
                "description": f"{split} split for defect detection",
                "contributor": "Mohamed Ali",
                "date_created": "2025-04-16"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            ],
            "images": imgs,
            "annotations": filter_annotations(imgs, coco_annotations),
            "categories": [{"id": 0, "name": "defect", "supercategory": "anomaly"}]
        }

        with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
            json.dump(split_ann, f, indent=2)
        print(f"{split.upper()} saved with {len(imgs)} images")
