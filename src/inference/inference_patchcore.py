import sys
import re
import cv2
import yaml
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
import shutil
import csv
import torch
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent))
from patchcore_inspection.src.patchcore.patchcore import PatchCore
from patchcore_inspection.src.patchcore import common
from patchcore_inspection.src.patchcore.datasets.mvtec import MVTecDataset

class PatchCoreInference:
    def __init__(self, config_path):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda:0")
        self.use_faiss_gpu = "cuda" in str(self.device)

        self.data_dir = (self.project_root / self.config["for_prediction"]).resolve()
        self.model_save_path = (self.project_root / self.config["output_dir_model_patchcore"]).resolve()
        self.output_path = (self.project_root / self.config["output_inference_dir"] / 'inference_patchcore').resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.battery_ng_counter = defaultdict(int)
        self.battery_bbox_rows = defaultdict(list)

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def patchcore_inference_with_mvtec_on_rois(self, model, original_image_path, roi_config, crop_output_dir, threshold=2.5, resize_size=256):
        original_img = cv2.imread(str(original_image_path))
        if original_img is None:
            return None, False

        overlay = original_img.copy()
        Path(crop_output_dir).mkdir(parents=True, exist_ok=True)
        crop_metadata = []

        stem = Path(original_image_path).stem
        for roi in roi_config:
            x_min, x_max = roi["x_min"], roi["x_max"]
            y_min, y_max = roi["y_min"], roi["y_max"]
            name = roi["name"]
            crop_img = original_img[y_min:y_max, x_min:x_max]
            crop_filename = f"{stem}_{name}_crop.png"
            crop_path = Path(crop_output_dir) / crop_filename
            cv2.imwrite(str(crop_path), crop_img)

            crop_metadata.append({
                "image_path": str(crop_path),
                "original_path": str(original_image_path),
                "crop_box": [x_min, y_min, x_max, y_max],
                "name": name,
                "crop_image": crop_img.copy()
            })

        dataset = MVTecDataset(source=crop_output_dir, resize=resize_size, imagesize=resize_size, inference_mode=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        anomaly_detected = False

        for idx, batch in enumerate(dataloader):
            image_tensor = batch["image"].to(model.device)
            scores, masks = model.predict_without_ground_truth([{"image": image_tensor}])
            score = scores[0]
            mask = masks[0]
            meta = crop_metadata[idx]
            crop_box = meta["crop_box"]
            crop_img = meta["crop_image"]
            crop_w = crop_box[2] - crop_box[0]
            crop_h = crop_box[3] - crop_box[1]

            norm_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            points = np.column_stack(np.where(norm_mask > np.percentile(norm_mask, 95)))
            if len(points) == 0:
                continue

            clustering = DBSCAN(eps=5, min_samples=5).fit(points)
            labels = clustering.labels_

            scale_x = crop_w / resize_size[1]
            scale_y = crop_h / resize_size[0]

            best_score = -1
            best_box = None
            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue
                cluster_points = points[labels == cluster_id]
                cluster_scores = [mask[y, x] for y, x in cluster_points]
                avg_score = np.mean(cluster_scores)

                if avg_score > best_score:
                    y1, x1 = np.min(cluster_points, axis=0)
                    y2, x2 = np.max(cluster_points, axis=0)
                    best_box = (x1, y1, x2, y2)
                    best_score = avg_score
            if best_box:
                x1, y1, x2, y2 = best_box
                abs_x1 = int(x1 * scale_x + crop_box[0])
                abs_y1 = int(y1 * scale_y + crop_box[1])
                abs_x2 = int(x2 * scale_x + crop_box[0])
                abs_y2 = int(y2 * scale_y + crop_box[1])

                if score > threshold:
                    anomaly_detected = True
                    filename_png = Path(original_image_path).with_suffix(".png").name
                    battery_id = Path(original_image_path).parent.parent.name
                    self.battery_bbox_rows[battery_id].append([
                        filename_png, abs_x1, abs_y1, abs_x2, abs_y2, 1, "", ""
                    ])
                    cv2.rectangle(overlay, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)

        return overlay, anomaly_detected

    def run_patchcore_on_filtered_images(self, model_path, base_folder, roi_config, crop_output_base, final_overlay_base, threshold=2.5, resize_size=(256, 256)):
        all_tif_paths = glob(f"{base_folder}/**/*.tif", recursive=True)
        filtered_paths = []

        for path in all_tif_paths:
            if "負極" not in path or "Z軸" not in path:
                continue
            match = re.search(r"_(\d{4})\.tif$", path)
            if match and 215 <= int(match.group(1)) <= 222:
                filtered_paths.append(path)

        model = PatchCore(device=self.device)
        model.load_from_path(
            str(model_path),
            device=self.device,
            nn_method=common.FaissNN(on_gpu=self.use_faiss_gpu, num_workers=32)
        )

        for image_path in tqdm(sorted(filtered_paths), desc="Processing 負極 Z軸 0215–0222"):
            image_path = Path(image_path)
            image_stem = image_path.stem
            battery_id = image_path.parent.parent.name
            axis = image_path.parent.name
            crop_output_dir = Path(crop_output_base) / image_stem

            overlay, is_anomaly = self.patchcore_inference_with_mvtec_on_rois(
                model, image_path, roi_config, crop_output_dir, threshold, resize_size
            )

            if is_anomaly:
                self.battery_ng_counter[battery_id] += 1

            if overlay is None:
                continue

            label = "anomaly" if is_anomaly else "normal"
            output_dir = Path(final_overlay_base) / label / battery_id / axis
            output_dir.mkdir(parents=True, exist_ok=True)
            final_overlay_path = output_dir / f"{image_stem}.png"
            cv2.imwrite(str(final_overlay_path), overlay)

        if Path(crop_output_base).exists():
            shutil.rmtree(crop_output_base, ignore_errors=True)

    def save_csv_results(self):
        ai_results_root = self.output_path / "AI_RESULTS"
        ai_results_root.mkdir(parents=True, exist_ok=True)

        for battery_id, rows in self.battery_bbox_rows.items():
            ng_count = self.battery_ng_counter[battery_id]
            if ng_count == 0:
                continue

            battery_result_dir = ai_results_root / battery_id
            battery_result_dir.mkdir(parents=True, exist_ok=True)

            log_path = battery_result_dir / f"{battery_id}_log.csv"
            with open(log_path, "w", newline="", encoding="utf-8") as f_log:
                writer = csv.writer(f_log)
                writer.writerow(["Name", "Result", "NG_countL", "NG_countP", "NG_countE"])
                seen = set()
                for row in rows:
                    if row[0] not in seen:
                        writer.writerow([row[0], "NG", 1, "", ""])
                        seen.add(row[0])

            detected_path = battery_result_dir / f"{battery_id}_detected.csv"
            with open(detected_path, "w", newline="", encoding="utf-8") as f_det:
                writer = csv.writer(f_det)
                writer.writerow(["file_name", "xmin", "ymin", "xmax", "ymax", "L", "P", "E"])
                for row in rows:
                    writer.writerow(row)

if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    inference = PatchCoreInference(config_path=config_path)

    for data_dir in [inference.data_dir]:
        inference.run_patchcore_on_filtered_images(
            model_path=inference.model_save_path,
            base_folder=data_dir,
            roi_config=[
                {"x_min": 127, "x_max": 294, "y_min": 448, "y_max": 576, "name": "left"},
                {"x_min": 733, "x_max": 900, "y_min": 448, "y_max": 576, "name": "right"},
            ],
            crop_output_base=(inference.output_path / 'patchcore_crops').resolve(),
            final_overlay_base=inference.output_path,
            threshold=2.5,
            resize_size=(168, 128),
        )

    if inference.config.get("export_ai_csv_files"):
        inference.save_csv_results()