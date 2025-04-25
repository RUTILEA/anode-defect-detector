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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from patchcore_inspection.src.patchcore.patchcore import PatchCore
from patchcore_inspection.src.patchcore import common
from patchcore_inspection.src.patchcore.datasets.mvtec import MVTecDataset
import torch
# === Setup ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

config_path = PROJECT_ROOT / "config.yaml"
print(f"📁 Loading config from: {config_path}")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

device= torch.device("cuda:0")
use_faiss_gpu = "cuda" in str(device)
print(f"🔍 FAISS is set to use: {'GPU' if use_faiss_gpu else 'CPU'}")

data_dir_positif = (PROJECT_ROOT / cfg["evaluation_patchcore_true_positif"]).resolve()
data_dir_negatif = (PROJECT_ROOT / cfg["evaluation_patchcore_true_negatif"]).resolve()
model_save_path = (PROJECT_ROOT / cfg["output_dir_model_patchcore"]).resolve()
output_path = (PROJECT_ROOT / cfg["output_inference_dir"] / 'patchcore').resolve()
output_path.mkdir(parents=True, exist_ok=True)


def patchcore_inference_with_mvtec_on_rois(
    model,
    original_image_path,
    roi_config,
    crop_output_dir,
    threshold=2.5,
    resize_size=256,
):
    original_img = cv2.imread(str(original_image_path))
    if original_img is None:
        return None, False

    overlay = original_img.copy()
    Path(crop_output_dir).mkdir(parents=True, exist_ok=True)
    crop_metadata = []

    for roi in roi_config:
        x_min, x_max = roi["x_min"], roi["x_max"]
        y_min, y_max = roi["y_min"], roi["y_max"]
        name = roi["name"]
        try:
            crop_img = original_img[y_min:y_max, x_min:x_max]
            crop_filename = f"{Path(original_image_path).stem}_{name}_crop.png"
            crop_path = Path(crop_output_dir) / crop_filename
            cv2.imwrite(str(crop_path), crop_img)
        except Exception as e:
            print(f"❌ Error cropping image: {e}")
            continue

        crop_metadata.append({
            "image_path": str(crop_path),
            "original_path": str(original_image_path),
            "crop_box": [x_min, y_min, x_max, y_max],
            "name": name,
            "crop_image": crop_img.copy()
        })

    dataset = MVTecDataset(
        source=crop_output_dir,
        resize=resize_size,
        imagesize=resize_size,
        inference_mode=True
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    save_crop_dir = Path(crop_output_dir) / "inference"
    save_crop_dir.mkdir(parents=True, exist_ok=True)

    anomaly_detected = False

    for idx, batch in enumerate(dataloader):
        image_tensor = batch["image"].to(model.device)
        scores, masks = model.predict_without_ground_truth([{"image": image_tensor}])
        score = scores[0]
        mask = masks[0]
        meta = crop_metadata[idx]

        crop_box = meta["crop_box"]
        crop_img = meta["crop_image"]
        crop_name = Path(meta["image_path"]).stem

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

        roi_overlay = crop_img.copy()

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

            rel_x1 = int(x1 * scale_x)
            rel_y1 = int(y1 * scale_y)
            rel_x2 = int(x2 * scale_x)
            rel_y2 = int(y2 * scale_y)

            if score > threshold:
                anomaly_detected = True
                cv2.rectangle(overlay, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                cv2.rectangle(roi_overlay, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 0, 255), 2)

        cv2.imwrite(str(save_crop_dir / f"{crop_name}_mask.png"), norm_mask)
        cv2.imwrite(str(save_crop_dir / f"{crop_name}_overlay.png"), roi_overlay)

    return overlay, anomaly_detected


def run_patchcore_on_filtered_images(
    model_path,
    base_folder,
    roi_config,
    crop_output_base,
    final_overlay_base,
    threshold=2.5,
    resize_size=256,
    device="cuda"
):
    all_tif_paths = glob(f"{base_folder}/**/*.tif", recursive=True)
    filtered_paths = []

    for path in all_tif_paths:
        if "負極" not in path or "Z軸" not in path:
            continue
        match = re.search(r"_(\d{4})\.tif$", path)
        if match:
            index = int(match.group(1))
            if 215 <= index <= 222:
                filtered_paths.append(path)

    model = PatchCore(device=device)
    model.load_from_path(
        str(model_path),
        device=device,
        nn_method=common.FaissNN(on_gpu="cuda" in str(device), num_workers=32)
    )

    for image_path in tqdm(sorted(filtered_paths), desc="🔁 Processing 負極 Z軸 0215–0222"):
        image_path = Path(image_path)
        image_stem = image_path.stem
        battery_id = image_path.parent.parent.name
        axis = image_path.parent.name

        crop_output_dir = Path(crop_output_base) / image_stem

        overlay, is_anomaly = patchcore_inference_with_mvtec_on_rois(
            model=model,
            original_image_path=image_path,
            roi_config=roi_config,
            crop_output_dir=crop_output_dir,
            threshold=threshold,
            resize_size=resize_size,
        )

        if overlay is None:
            print(f"❌ Skipping {image_path} due to read error")
            continue

        label = "anomaly" if is_anomaly else "normal"
        output_dir = Path(final_overlay_base) / label / battery_id / axis 
        output_dir.mkdir(parents=True, exist_ok=True)

        final_overlay_path = output_dir / f"{image_stem}_overlay.png"
        cv2.imwrite(str(final_overlay_path), overlay)
    if Path(crop_output_base).exists():
        try:
            shutil.rmtree(crop_output_base)
        except Exception as e:
            print(f"❌ Could not delete {crop_output_base}: {e}")
                
    return f"✅ Done! Processed {len(filtered_paths)} 負極 Z軸 images with index 215–222."


for data_dir in [data_dir_positif, data_dir_negatif]:
    run_patchcore_on_filtered_images(
        model_path=model_save_path,
        base_folder=data_dir,
        roi_config=[
            {"x_min": 127, "x_max": 294, "y_min": 448, "y_max": 576, "name": "left"},
            {"x_min": 733, "x_max": 900, "y_min": 448, "y_max": 576, "name": "right"},
        ],
        crop_output_base=(output_path / 'patchcore_crops').resolve(),
        final_overlay_base=output_path,
        threshold=2.5,
        resize_size=(168, 128),
        device=device
    )

