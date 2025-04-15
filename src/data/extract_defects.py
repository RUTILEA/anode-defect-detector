import os
import cv2
import json
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
def extract_defects_from_annotations(annotations_root, images_root, output_dir, roi_config):
    defects = []
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    os.makedirs(output_dir, exist_ok=True)

    for folder in sorted(os.listdir(annotations_root))[:4]:
        ann_dir = os.path.join(annotations_root, folder)
        print(f"🔍 Processing annotation folder: {folder}")
        json_file = next((f for f in os.listdir(ann_dir) if f.endswith(".json")), None)
        if not json_file:
            print(f"⚠️ No JSON file in {ann_dir}")
            continue

        with open(os.path.join(ann_dir, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        for file, content in data.items():
            regions = content.get("regions", {})
            if not regions:
                print(f"⚠️ No regions in file: {file}")
                continue

            for region in regions.values():
                if region["shape_attributes"]["name"] != "polygon":
                    print(f"⚠️ Region in {file} is not a polygon")
                    continue

                points_x = region["shape_attributes"]["all_points_x"]
                points_y = region["shape_attributes"]["all_points_y"]
                pts = np.array([(int(x), int(y)) for x, y in zip(points_x, points_y)], dtype=np.int32)

                battery_id_folder = os.path.basename(ann_dir)
                matched_folder = None

                for root, dirs, _ in os.walk(images_root):
                    for d in dirs:
                        if battery_id_folder in d:
                            matched_folder = os.path.join(root, d)
                            break
                    if matched_folder:
                        break

                if not matched_folder:
                    print(f"⚠️ No matching image folder for battery ID: {battery_id_folder}")
                    continue

                image_candidates = [f for f in os.listdir(matched_folder) if f.endswith(".png")]
                base_key = "_".join(file.split("_")[-2:])
                matched_file = next((f for f in image_candidates if base_key in f), None)
                if not matched_file:
                    print(f"⚠️ No matching image file for key: {base_key} in {matched_folder}")
                    continue

                full_path = os.path.join(matched_folder, matched_file)
                img = cv2.imread(full_path)
                if img is None:
                    print(f"⚠️ Failed to read image: {full_path}")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                x, y, w, h = cv2.boundingRect(pts)
                patch = img_rgb[y:y+h, x:x+w]
                patch_mask = mask[y:y+h, x:x+w] > 0

                for roi in roi_config:
                    if not (x + w < roi["x_min"] or x > roi["x_max"] or y + h < roi["y_min"] or y > roi["y_max"]):
                        roi_crop = img_rgb[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
                        file_base = os.path.splitext(os.path.basename(file))[0].replace(":", "").replace(" ", "").replace("/", "").replace("\\", "")
                        crop_filename = f"{battery_id_folder}_{file_base}_{roi['name']}_crop_original.png"

                        crop_path = os.path.join(output_dir, crop_filename)
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
                            "battery_id": battery_id_folder
                        })

                        image_id += 1
                        annotation_id += 1
                        break

    print(f"\n🧩 Extracted {len(defects)} defects")
    print(f"🧩 Extracted {len(images)} ROI crops")
    return defects, images, annotations, image_id, annotation_id
