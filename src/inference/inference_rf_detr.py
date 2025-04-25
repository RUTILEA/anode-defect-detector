# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from pathlib import Path
# import supervision as sv
# import yaml
# import sys
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🔍 Using device: {device}")
# from rfdetr import RFDETRBase
# import torch


# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.append(str(PROJECT_ROOT))

# config_path = PROJECT_ROOT / "config.yaml"
# print(f"📁 Loading config from: {config_path}")
# with open(config_path, "r") as f:
#     cfg = yaml.safe_load(f)

# checkpoint_path = PROJECT_ROOT / ".." / cfg["output_dir_model_rf_detr"] / "checkpoint_best_ema.pth"
# ANODE_DIR = PROJECT_ROOT / ".." / cfg["good_images_dir"]
# ANNODE_DIR2
# OUTPUT_DIR = PROJECT_ROOT / cfg["output_inference_dir"] /  'inference_rf_detr'

# model = RFDETRBase(pretrain_weights=str(checkpoint_path), num_classes=1,device='cuda')

# class_labels = ['upper defect']
# THRESHOLD = 0.7

# roi_config = [
#     {"x_min": 127, "x_max": 294, "y_min": 448, "y_max": 576, "name": "left"},
#     {"x_min": 733, "x_max": 900, "y_min": 448, "y_max": 576, "name": "right"},
# ]

# bbox_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=2, smart_position=True)

# image_paths = []
# for dataset_dir in [ANODE_DIR, ANNODE_DIR2]:
#     for battery_folder in dataset_dir.iterdir():
#         z_axis_folder = battery_folder / "[Z軸]"
#         if not z_axis_folder.exists():
#             continue

#         for tif_path in z_axis_folder.glob("*.tif"):
#             name = tif_path.name
#             if "負極" not in name:
#                 continue
#             try:
#                 index = int(name.split("_")[-1].replace(".tif", ""))
#                 if 215 <= index <= 222:
#                     image_paths.append(tif_path)
#             except ValueError:
#                 continue

# print(f"📷 Found {len(image_paths)} images for inference.")

# for image_path in tqdm(image_paths, desc="RF-DETR Inference"):
#     image_np = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
#     if image_np is None:
#         print(f"❌ Failed to read: {image_path}")
#         continue

#     image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#     image_pil = Image.fromarray(image_rgb)
#     detections_all = []

#     for roi in roi_config:
#         roi_crop = image_rgb[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
#         roi_image = Image.fromarray(roi_crop)

#         detections = model.predict(roi_image, threshold=THRESHOLD)
#         if len(detections.xyxy) > 0:
#             detections.xyxy[:, [0, 2]] += roi["x_min"]
#             detections.xyxy[:, [1, 3]] += roi["y_min"]
#             detections_all.append(detections)

#     battery_id = image_path.parents[1].name
#     category_dir = "anomaly" if detections_all else "normal"
#     save_dir = OUTPUT_DIR / category_dir / battery_id
#     save_dir.mkdir(parents=True, exist_ok=True)
#     save_path = save_dir / f"{image_path.stem}.png"

#     if detections_all:
#         detections_merged = sv.Detections.merge(detections_all)
#         labels = [
#             f"{class_labels[class_id]} {confidence:.2f}"
#             for class_id, confidence in zip(detections_merged.class_id, detections_merged.confidence)
#         ]
#         annotated_image = bbox_annotator.annotate(image_pil.copy(), detections_merged)
#         annotated_image = label_annotator.annotate(annotated_image, detections_merged, labels)
#         annotated_image.save(save_path)
#     else:
#         image_pil.save(save_path)

# print(f"✅ Done. Saved to {OUTPUT_DIR/'ok'} and {OUTPUT_DIR/'anomaly'}")





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
from rfdetr import RFDETRBase

def run_rfdetr_inference(dataset_dirs, output_dir, checkpoint_path, num_classes=1, threshold=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")

    model = RFDETRBase(pretrain_weights=str(checkpoint_path), num_classes=num_classes)
    model.model.model.to(device)
    model.model.device = device
    model.model.model.eval()

    class_labels = ['upper defect']
    roi_config = [
        {"x_min": 127, "x_max": 294, "y_min": 448, "y_max": 576, "name": "left"},
        {"x_min": 733, "x_max": 900, "y_min": 448, "y_max": 576, "name": "right"},
    ]

    bbox_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=2, smart_position=True)

    # === Collect Images ===
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
                    if 215 <= index <= 222:
                        image_paths.append(tif_path)
                except ValueError:
                    continue

    print(f"📷 Found {len(image_paths)} images for inference.")

    # === Inference Loop ===
    for image_path in tqdm(image_paths, desc="RF-DETR Inference"):
        image_np = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_np is None:
            print(f"❌ Failed to read: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        detections_all = []

        for roi in roi_config:
            roi_crop = image_rgb[roi["y_min"]:roi["y_max"], roi["x_min"]:roi["x_max"]]
            roi_image = Image.fromarray(roi_crop)
            detections = model.predict(roi_image, threshold=threshold)

            if len(detections.xyxy) > 0:
                detections.xyxy[:, [0, 2]] += roi["x_min"]
                detections.xyxy[:, [1, 3]] += roi["y_min"]
                detections_all.append(detections)

        battery_id = image_path.parents[1].name
        category_dir = "anomaly" if detections_all else "normal"
        dataset_name = Path(image_path).parts[-4]  # Extract top-level dataset folder name
        save_dir = Path(output_dir)  / category_dir / battery_id
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{image_path.stem}.png"

        if detections_all:
            detections_merged = sv.Detections.merge(detections_all)
            labels = [
                f"{class_labels[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections_merged.class_id, detections_merged.confidence)
            ]
            annotated_image = bbox_annotator.annotate(image_pil.copy(), detections_merged)
            annotated_image = label_annotator.annotate(annotated_image, detections_merged, labels)
            annotated_image.save(save_path)
        else:
            image_pil.save(save_path)

    print(f"✅ Done. Saved to {output_dir}")


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))

    config_path = PROJECT_ROOT / "config.yaml"
    print(f"📁 Loading config from: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        

    checkpoint_path = PROJECT_ROOT / ".." / cfg["output_dir_model_rf_detr"] / "checkpoint_best_ema.pth"
    dataset1 = PROJECT_ROOT  / cfg["evaluation_patchcore_true_negatif"]
    dataset2 = PROJECT_ROOT  / cfg["evaluation_patchcore_true_positif"]
    output_dir = PROJECT_ROOT / cfg["output_inference_dir"] /  'inference_rf_detr'
    

    run_rfdetr_inference(
        dataset_dirs=[dataset1, dataset2],
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        num_classes=1,
        threshold=0.7
    )
