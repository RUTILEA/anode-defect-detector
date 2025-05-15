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
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        self.data_dir = (self.project_root.parent / self.config["input_dir"]).resolve()
        self.model_save_path = (self.project_root.parent / self.config["output_dir_model_patchcore"]).resolve()
        self.output_path = (self.project_root.parent / self.config["output_inference_dir"] / 'inference_patchcore').resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.min_idx = self.config.get("filter_index_range").get("min")
        self.max_idx = self.config.get("filter_index_range").get("max")
        self.class_label = self.config.get("class_label")
        self.battery_sequences = defaultdict(lambda: {
            'current_sequence': deque(maxlen=10),
            'last_frame_num': None,
            'pending_overlays': []
        })        
        self.battery_bbox_rows = defaultdict(list)
        self.consecutive_threshold = 4

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def extract_frame_number(self, path):
        match = re.search(r"_(\d{4})\.tif$", str(path))
        return int(match.group(1)) if match else None

    def run_patchcore_on_filtered_images(self, model_path, base_folder, roi_config, crop_output_base, final_overlay_base, threshold=2.5, resize_size=(256, 256)):
        all_tif_paths = glob(f"{base_folder}/**/*.tif", recursive=True)
        print(f"Found {len(all_tif_paths)} TIF files in {base_folder}")
        path_data = []
        for path in all_tif_paths:
            if "負極" not in path or "Z軸" not in path:
                continue
            frame_num = self.extract_frame_number(path)
            print(f"Processing {path} with frame number {frame_num}")
            if frame_num and self.min_idx <= frame_num <= self.max_idx:
                battery_id = Path(path).parent.parent.name
                path_data.append({
                    'path': path,
                    'frame_num': frame_num,
                    'battery_id': battery_id
                })

        path_data.sort(key=lambda x: (x['battery_id'], x['frame_num']))

        model = PatchCore(device=self.device)
        model.load_from_path(
            str(model_path),
            device=self.device,
            nn_method=common.FaissNN(on_gpu=self.use_faiss_gpu, num_workers=32)
        )

        for data in tqdm(path_data, desc="PATCHCORE Inference"):
            image_path = Path(data['path'])
            frame_num = data['frame_num']
            battery_id = data['battery_id']
            image_stem = image_path.stem
            axis = image_path.parent.name
            crop_output_dir = Path(crop_output_base) / image_stem

            overlay, bbox_drawn, score, bbox = self.patchcore_inference_with_mvtec_on_rois(
                model, image_path, roi_config, crop_output_dir, threshold, resize_size
            )

            seq_tracker = self.battery_sequences[battery_id]
            if 'valid_frames' not in seq_tracker:
                seq_tracker['valid_frames'] = []
            seq_tracker['valid_frames'].append(frame_num)
            seq_tracker['valid_frames'].sort()

            idx = seq_tracker['valid_frames'].index(frame_num)
            previous_frame = seq_tracker['valid_frames'][idx - 1] if idx > 0 else None
            is_consecutive = (previous_frame == seq_tracker['last_frame_num'])

            if bbox_drawn and score > threshold:
                if is_consecutive or not seq_tracker['current_sequence']:
                    seq_tracker['current_sequence'].append(frame_num)
                else:
                    seq_tracker['current_sequence'] = deque([frame_num], maxlen=self.consecutive_threshold)
            else:
                seq_tracker['current_sequence'].clear()

            seq_tracker['last_frame_num'] = frame_num

            has_consecutive_anomalies = (
                len(seq_tracker['current_sequence']) >= self.consecutive_threshold and
                all(seq_tracker['current_sequence'][i] + 1 == seq_tracker['current_sequence'][i + 1]
                    for i in range(len(seq_tracker['current_sequence']) - 1))
            )

            if overlay is None:
                continue

            if bbox_drawn and score > threshold:
                seq_tracker['pending_overlays'].append({
                    'overlay': overlay,
                    'bbox': bbox,
                    'image_path': image_path,
                    'image_stem': image_stem,
                    'axis': axis,
                    'battery_id': battery_id
                })

            if has_consecutive_anomalies:
                for item in seq_tracker['pending_overlays']:
                    output_dir = Path(final_overlay_base) / "anomaly" / item['battery_id'] / item['axis']
                    output_dir.mkdir(parents=True, exist_ok=True)
                    final_overlay_path = output_dir / f"{item['image_stem']}.png"
                    cv2.imwrite(str(final_overlay_path), item['overlay'])

                    x1, y1, x2, y2 = item['bbox']
                    filename_png = item['image_path'].with_suffix(".png").name
                    self.battery_bbox_rows[item['battery_id']].append([
                        filename_png, x1, y1, x2, y2, 1, "", ""
                    ])

                seq_tracker['pending_overlays'] = []
                seq_tracker['current_sequence'].clear()
            else:
                if not has_consecutive_anomalies and overlay is not None:
                    output_dir = Path(final_overlay_base) / "normal" / battery_id / axis
                    output_dir.mkdir(parents=True, exist_ok=True)
                    final_overlay_path = output_dir / f"{image_stem}.png"
                    cv2.imwrite(str(final_overlay_path), cv2.imread(str(image_path)))

        if Path(crop_output_base).exists():
            shutil.rmtree(crop_output_base, ignore_errors=True)

    def patchcore_inference_with_mvtec_on_rois(self, model, original_image_path, roi_config, crop_output_dir, threshold=2.5, resize_size=256):
        original_img = cv2.imread(str(original_image_path))
        if original_img is None:
            return None, False, 0, None

        overlay = original_img.copy()
        Path(crop_output_dir).mkdir(parents=True, exist_ok=True)
        crop_metadata = []

        stem = Path(original_image_path).stem
        battery_id = Path(original_image_path).parent.parent.name

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

        bbox_drawn = False
        max_score = 0
        final_bbox = None

        for idx, batch in enumerate(dataloader):
            image_tensor = batch["image"].to(model.device)
            scores, masks = model.predict_without_ground_truth([{"image": image_tensor}])
            score = scores[0]
            mask = masks[0]

            if score > max_score:
                max_score = score

            if score <= threshold:
                continue

            meta = crop_metadata[idx]
            crop_box = meta["crop_box"]
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

                final_bbox = (abs_x1, abs_y1, abs_x2, abs_y2)
                bbox_drawn = True
                cv2.rectangle(overlay, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                text_origin = (abs_x1, abs_y1 - 5 if abs_y1 - 5 > 10 else abs_y1 + 15)
                cv2.putText(
                    overlay, self.class_label, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA
                )

        return overlay, bbox_drawn, max_score, final_bbox

    def save_csv_results(self):
        ai_results_root = self.output_path / "AI_RESULTS"
        ai_results_root.mkdir(parents=True, exist_ok=True)

        for battery_id, rows in self.battery_bbox_rows.items():
            if not rows:
                continue

            battery_result_dir = ai_results_root / battery_id
            battery_result_dir.mkdir(parents=True, exist_ok=True)
            sanitized_id = battery_id.replace("/", "_").replace("[", "").replace("]", "")

            log_path = battery_result_dir / f"{sanitized_id}_log.csv"
            with open(log_path, mode="w", newline="", encoding="utf-8") as f_log:
                writer = csv.writer(f_log)
                writer.writerow(["Name", "Result", "NG_countL", "NG_countP", "NG_countE"])
                seen = set()
                for row in rows:
                    if row[0] not in seen:
                        writer.writerow([row[0], "NG", 1, "", ""])
                        seen.add(row[0])

            detected_path = battery_result_dir / f"{sanitized_id}_detected.csv"
            with open(detected_path, mode="w", newline="", encoding="utf-8") as f_det:
                writer = csv.writer(f_det)
                writer.writerow(["file_name", "xmin", "ymin", "xmax", "ymax", "L", "P", "E"])
                for row in rows:
                    writer.writerow(row)
