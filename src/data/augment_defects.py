import os
import cv2
import json
import math
import shutil
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


class DefectAugmentor:
    def __init__(self, roi_config):
        self.roi_config = roi_config

    @staticmethod
    def enhance_contrast_patch(base_img, mask, patch, y, x, strength=0.6):
        region = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)[y:y+patch.shape[0], x:x+patch.shape[1]]
        region_median = np.median(region[mask])
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        patch_median = np.median(patch_gray[mask])
        shift = int(np.clip((region_median - patch_median) * strength, -80, 80))
        adjusted = patch.astype(np.int16)
        adjusted[mask] += shift
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def detect_dynamic_circles(gray_img, min_radius=12, max_radius=20):
        circles = cv2.HoughCircles(
            gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=22, minRadius=min_radius, maxRadius=max_radius
        )
        return [
            {"x": int(x), "y": int(y), "radius": int(r)}
            for x, y, r in circles[0]
        ] if circles is not None else []

    @staticmethod
    def box_circle_overlap(box, circle, buffer=0):
        x_min, y_min, w, h = box
        x_max, y_max = x_min + w, y_min + h
        cx, cy, cr = circle["x"], circle["y"], circle["radius"] + buffer
        closest_x = max(x_min, min(cx, x_max))
        closest_y = max(y_min, min(cy, y_max))
        distance = math.hypot(closest_x - cx, closest_y - cy)
        return distance < cr

    def generate_half_positions(self, roi, half, w, h, min_dist, y_split, margin, buffer, circles, existing_boxes):
        y_min, y_max = (roi["y_min"], y_split - margin) if half == "top" else (y_split + margin, roi["y_max"])
        tries = 0
        while tries < 20:
            x = random.randint(roi["x_min"] + w//2 + margin, roi["x_max"] - w//2 - margin)
            y = random.randint(y_min + h//2, y_max - h//2)
            box = [x - w//2, y - h//2, w, h]
            overlaps = any(
                abs(box[0] - eb[0]) < min_dist and abs(box[1] - eb[1]) < min_dist for eb in existing_boxes
            )
            circle_conflict = any(self.box_circle_overlap(box, c, buffer=buffer) for c in circles)
            if not overlaps and not circle_conflict:
                return {"x": x, "y": y}
            tries += 1
        return None

    @staticmethod
    def convert_tif_images_to_png(input_root, output_root):
        input_root = Path(input_root)
        output_root = Path(output_root)
        count = 0
        for root, _, files in os.walk(input_root):
            for file in files:
                if (file.lower().endswith(".tif") and "負極" in file and "Z軸" in root and
                    not str(Path(root)).startswith(str(output_root)) and
                    any(f"_{i:04}" in file for i in range(215, 223))):
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(input_root).with_suffix(".png")
                    output_path = output_root / rel_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        img = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            cv2.imwrite(str(output_path), img)
                            count += 1
                    except Exception as e:
                        print(f"Error converting {full_path}: {e}")

    def augment_images_and_generate_coco(self, defects, good_images_dir, dest_good_images_dir, output_dir, img_id_start, ann_id_start, dynamic_circle_detection=True, placements_per_image=10):
        self.convert_tif_images_to_png(good_images_dir, dest_good_images_dir)
        image_id = img_id_start
        annotation_id = ann_id_start
        images, annotations = [], []
        total_saved = 0
        good_images = sorted([str(p) for p in Path(dest_good_images_dir).rglob("*.png")])
        for img_path in tqdm(good_images, desc="Augmenting"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            base_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            base_name = Path(img_path).stem
            gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            for roi in self.roi_config:
                detected_circles = self.detect_dynamic_circles(blurred) if dynamic_circle_detection else []
                selected_defects = random.sample(defects, min(placements_per_image, len(defects)))
                placed_boxes = []
                for defect_idx, selected_defect in enumerate(selected_defects):
                    pos = self.generate_half_positions(
                        roi, random.choice(["top", "bottom"]),
                        selected_defect["w"], selected_defect["h"],
                        min_dist=12, y_split=roi["circles_y"][0],
                        margin=3, buffer=7,
                        circles=detected_circles, existing_boxes=placed_boxes
                    )
                    if not pos:
                        continue
                    x = pos["x"] - selected_defect["w"] // 2
                    y = pos["y"] - selected_defect["h"] // 2
                    placed_boxes.append((x, y))
                    patch = self.enhance_contrast_patch(base_rgb, selected_defect["mask"], selected_defect["patch"], y, x)
                    for c in range(3):
                        base_rgb[y:y+selected_defect["h"], x:x+selected_defect["w"], c][selected_defect["mask"]] = patch[..., c][selected_defect["mask"]]
                    crop = base_rgb[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
                    filename = f"{base_name}_{roi['name']}_d{defect_idx}.png"
                    cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    images.append({"id": image_id, "file_name": filename, "width": crop.shape[1], "height": crop.shape[0]})
                    annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 0, "bbox": [x - roi["x_min"], y - roi["y_min"], selected_defect["w"], selected_defect["h"]], "area": selected_defect["w"] * selected_defect["h"], "iscrowd": 0, "segmentation": []})
                    image_id += 1
                    annotation_id += 1
                    total_saved += 1
        return images, annotations, total_saved
    
    def crop_and_save_rois(self, input_dir, output_dir):
        input_root = Path(input_dir)
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        count = 0

        for root, _, files in os.walk(input_root):
            for file in files:
                if (
                    file.lower().endswith(".tif")
                    and "負極" in file
                    and "Z軸" in root
                    and not str(Path(root)).startswith(str(output_root))
                    and any(f"_{i:04}" in file for i in range(160, 240))
                ):
                    root_path = Path(root)
                    full_path = root_path / file
                    try:
                        img = cv2.imread(str(full_path), cv2.IMREAD_UNCHANGED)
                        if img is None:
                            continue
                        base_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        base_name = Path(file).stem

                        for roi in self.roi_config:
                            crop = base_rgb[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
                            roi_name = roi.get("name", "roi")
                            filename = f"{base_name}_{roi_name}_crop.png"
                            save_path = output_root / filename
                            cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                            count += 1
                    except Exception as e:
                        print(f"❌ Error processing {full_path}: {e}")

        print(f"Total cropped ROI images saved: {count}")
