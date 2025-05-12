import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import supervision as sv
import yaml
import sys
import csv
from collections import defaultdict
from rfdetr import RFDETRBase
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RFDETRInference:
    def __init__(self, config_path):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_name = self.config["rf_detr_checkpoint"]
        self.model = RFDETRBase(
            pretrain_weights=str(self.project_root / ".." / self.config["output_dir_model_rf_detr"] / self.checkpoint_name),
            num_classes=1,pretrained=False,
        )
        self.model.model.model.to(self.device)
        self.model.model.device = self.device
        self.model.model.model.eval()
        self.class_label = [self.config["class_label"]]
        self.roi_config =  self.config.get("roi_config")
        self.min_idx = self.config.get("filter_index_range").get("min")
        self.max_idx = self.config.get("filter_index_range").get("max")
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=2, smart_position=True)
        self.bbox_annotator = sv.BoxAnnotator()
        self.battery_ng_counter = defaultdict(int)
        self.battery_bbox_rows = defaultdict(list)

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def collect_image_paths(self, dataset_dirs):
        image_paths = []
        for dataset_dir in dataset_dirs:
            for battery_folder in Path(dataset_dir).iterdir():
                z_axis_folder = battery_folder / "[Z軸]"
                if not z_axis_folder.exists():
                    continue
                for tif_path in z_axis_folder.glob("*.tif"):
                    name = tif_path.name
                    if "負極" not in name:
                        continue
                    try:
                        index = int(name.split("_")[-1].replace(".tif", ""))
                        if self.min_idx <= index <= self.max_idx:
                            image_paths.append(tif_path)
                    except ValueError:
                        continue
        return image_paths

    def run_inference(self, dataset_dirs, output_dir, threshold=0.8):
        image_paths = self.collect_image_paths(dataset_dirs)

        for image_path in tqdm(image_paths, desc="RF-DETR Inference"):
            image_np = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_np is None:
                continue

            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            detections_all = []

            for roi in self.roi_config:
                roi_crop = image_rgb[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
                roi_image = Image.fromarray(roi_crop)
                detections = self.model.predict(roi_image, threshold=threshold)
                if len(detections.xyxy) > 0:
                    detections.xyxy[:, [0, 2]] += roi["x_min"]
                    detections.xyxy[:, [1, 3]] += roi["y_min"]
                    detections_all.append(detections)

            battery_folder = image_path.parents[1].name 
            z_axis_folder = image_path.parents[0].name
            battery_id = os.path.join(battery_folder, z_axis_folder)

            if detections_all:
                self.battery_ng_counter[battery_id] += 1

            category_dir = "anomaly" if detections_all else "normal"
            save_dir = Path(output_dir) / category_dir / battery_id
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{image_path.stem}.png"

            if detections_all:
                detections_merged = sv.Detections.merge(detections_all)
                labels = [
                    f"{self.class_label[class_id]} {confidence:.2f}"
                    for class_id, confidence in zip(detections_merged.class_id, detections_merged.confidence)
                ]
                annotated_image = self.bbox_annotator.annotate(image_pil.copy(), detections_merged)
                annotated_image = self.label_annotator.annotate(annotated_image, detections_merged, labels)
                annotated_image.save(save_path)

                filename_png = image_path.with_suffix(".png").name
                for box in detections_merged.xyxy:
                    xmin, ymin, xmax, ymax = map(int, box.tolist())
                    self.battery_bbox_rows[battery_id].append([filename_png, xmin, ymin, xmax, ymax, 1, "", ""])
            else:
                image_pil.save(save_path)

        if self.config.get("export_ai_csv_files"):
            self.save_csv_results(output_dir)

    def save_csv_results(self, output_dir):
        ai_results_root = Path(output_dir) / "AI_RESULTS"
        ai_results_root.mkdir(parents=True, exist_ok=True)

        for battery_id, rows in self.battery_bbox_rows.items():
            ng_count = self.battery_ng_counter[battery_id]
            if ng_count == 0:
                continue

            battery_result_dir = ai_results_root / battery_id
            battery_result_dir.mkdir(parents=True, exist_ok=True)

            log_path = battery_result_dir / f"{battery_id}_log.csv"
            with open(log_path, mode="w", newline="", encoding="utf-8") as f_log:
                writer = csv.writer(f_log)
                writer.writerow(["Name", "Result", "NG_countL", "NG_countP", "NG_countE"])
                seen = set()
                for row in rows:
                    if row[0] not in seen:
                        writer.writerow([row[0], "NG", 1, "", ""])
                        seen.add(row[0])

            detected_path = battery_result_dir / f"{battery_id}_detected.csv"
            with open(detected_path, mode="w", newline="", encoding="utf-8") as f_det:
                writer = csv.writer(f_det)
                writer.writerow(["file_name", "xmin", "ymin", "xmax", "ymax", "L", "P", "E"])
                for row in rows:
                    writer.writerow(row)