import os
import cv2
import json
import numpy as np
from pathlib import Path

class DefectExtractor:
    def __init__(self, annotations_root, images_root, output_dir, roi_config):
        self.annotations_root = annotations_root
        self.images_root = images_root
        self.output_dir = output_dir
        self.roi_config = roi_config
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self):
        defects, images, annotations = [], [], []
        image_id, annotation_id = 1, 1

        for folder in sorted(os.listdir(self.annotations_root))[:4]:
            ann_dir = os.path.join(self.annotations_root, folder)
            json_file = next((f for f in os.listdir(ann_dir) if f.endswith(".json")), None)
            if not json_file:
                print(f"No JSON file in {ann_dir}")
                continue

            with open(os.path.join(ann_dir, json_file), "r", encoding="utf-8") as f:
                data = json.load(f)

            for file, content in data.items():
                regions = content.get("regions", {})
                if not regions:
                    print(f"No regions in file: {file}")
                    continue

                for region in regions.values():
                    if region["shape_attributes"]["name"] != "polygon":
                        print(f"Region in {file} is not a polygon")
                        continue

                    pts = np.array([
                        (int(x), int(y))
                        for x, y in zip(region["shape_attributes"]["all_points_x"],
                                        region["shape_attributes"]["all_points_y"])
                    ], dtype=np.int32)

                    battery_id = os.path.basename(ann_dir)
                    matched_folder = self._find_image_folder(battery_id)
                    if not matched_folder:
                        print(f"No matching image folder for battery ID: {battery_id}")
                        continue

                    matched_file = self._find_image_file(matched_folder, file)
                    if not matched_file:
                        print(f"No matching image file for key in {matched_folder}")
                        continue

                    full_path = os.path.join(matched_folder, matched_file)
                    img = cv2.imread(full_path)
                    if img is None:
                        print(f"Failed to read image: {full_path}")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)

                    x, y, w, h = cv2.boundingRect(pts)
                    patch = img_rgb[y:y+h, x:x+w]
                    patch_mask = mask[y:y+h, x:x+w] > 0

                    for roi in self.roi_config:
                        if x + w < roi["x_min"] or x > roi["x_max"] or y + h < roi["y_min"] or y > roi["y_max"]:
                            continue

                        roi_crop = img_rgb[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
                        file_base = os.path.splitext(os.path.basename(file))[0].replace(":", "").replace(" ", "").replace("/", "").replace("\\", "")
                        crop_filename = f"{battery_id}_{file_base}_{roi['name']}_crop_original.png"
                        crop_path = os.path.join(self.output_dir, crop_filename)
                        cv2.imwrite(crop_path, cv2.cvtColor(roi_crop, cv2.COLOR_RGB2BGR))

                        images.append({
                            "id": image_id,
                            "file_name": crop_filename,
                            "width": roi_crop.shape[1],
                            "height": roi_crop.shape[0]
                        })

                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x - roi["x_min"], y - roi["y_min"], w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })

                        defects.append({
                            "patch": patch,
                            "mask": patch_mask,
                            "w": w,
                            "h": h,
                            "battery_id": battery_id
                        })

                        image_id += 1
                        annotation_id += 1
                        break

        print(f"\n Extracted {len(defects)} defects")
        print(f" Extracted {len(images)} ROI crops")
        return defects, images, annotations, image_id, annotation_id

    def _find_image_folder(self, battery_id):
        for root, dirs, _ in os.walk(self.images_root):
            for d in dirs:
                if battery_id in d:
                    return os.path.join(root, d)
        return None

    def _find_image_file(self, folder, file_key):
        image_candidates = [f for f in os.listdir(folder) if f.endswith(".png")]
        base_key = "_".join(file_key.split("_")[-2:])
        return next((f for f in image_candidates if base_key in f), None)
