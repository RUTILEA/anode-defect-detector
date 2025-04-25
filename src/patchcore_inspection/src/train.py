# import os
# from pathlib import Path
# import torch
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import torchvision.transforms.functional as TF
# import csv
# import cv2
# import random

# from patchcore.patchcore import PatchCore
# from patchcore.datasets.mvtec import MVTecDataset, DatasetSplit
# from patchcore.backbones import load
# from torch.utils.data import Subset
# from sklearn.cluster import DBSCAN

# data_dir = Path("/var/lib/containerd/battery_project/anode-defect-detector/data/raw")
# classname = "patchcore_data"
# output_path = Path("/var/lib/containerd/battery_project/anode-defect-detector/results")
# output_path.mkdir(parents=True, exist_ok=True)
# device = torch.device("cuda:0")
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# backbone = load("wideresnet101")
# backbone.name = "wideresnet101"
# model = PatchCore(device=device)
# model.load(
#     backbone=backbone,
#     layers_to_extract_from=["layer2", "layer3"],
#     input_shape=(3, 256, 256),
#     target_embed_dimension=1024,
#     pretrain_embed_dimension=2048,
#     faiss_on_gpu=True,
#     device=device,
# )

# train_dataset = MVTecDataset(
#     source=data_dir,
#     classname=classname,
#     resize=256,
#     imagesize=256,
#     split=DatasetSplit.TRAIN,
# )
# total_indices = list(range(len(train_dataset)))
# subset_indices = random.sample(total_indices, k=min(3000, len(total_indices)))
# train_dataset = Subset(train_dataset, subset_indices)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# # === Train the model ===
# print("🚀 Training PatchCore...")
# model.fit(train_loader)
# print("✅ Training completed.")
# model_save_path = output_path / "patchcore_model"
# model_save_path.mkdir(exist_ok=True)
# model.save_to_path(str(model_save_path))
# print(f"✅ Model saved to {model_save_path}")

# === Load test dataset ===
# test_dataset = MVTecDataset(
#     source=data_dir,
#     classname=classname,
#     resize=256,
#     imagesize=256,
#     split=DatasetSplit.TEST
# )
# total_indices = list(range(len(test_dataset)))
# subset_indices = random.sample(total_indices, k=min(10, len(total_indices)))
# test_dataset = Subset(test_dataset, subset_indices)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# scores, masks  = model.predict_without_ground_truth(test_loader)

# with open(output_path / "prediction_scores2.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Index", "Score", "GroundTruth"])
#     for i, score in enumerate(scores):
#         writer.writerow([i, round(score, 4), "N/A"])


# === Save masks
# mask_dir = output_path / "predicted_masks2"
# mask_dir.mkdir(exist_ok=True)
# for idx, mask in enumerate(masks):
#     mask_img = Image.fromarray((mask * 255).astype(np.uint8))
#     mask_img.save(mask_dir / f"mask_{idx}.png")

# # === Utility: unnormalize
# def unnormalize(tensor):
#     mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
#     std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
#     return (tensor * std + mean).clamp(0, 1)

# for idx in range(len(test_dataset)):
#     image_tensor = test_dataset[idx]["image"]
#     image = unnormalize(image_tensor).permute(1, 2, 0).numpy()
#     image_bgr = (image * 255).astype(np.uint8)[:, :, ::-1]
#     pred_mask = masks[idx]

#     # === Normalize heatmap
#     norm_mask = cv2.normalize(pred_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     heatmap = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

#     # === Compute dynamic threshold using 95th percentile
#     threshold = np.percentile(norm_mask, 95)
#     points = np.column_stack(np.where(norm_mask > threshold))
#     overlay_with_box = image_bgr.copy()

#     if len(points) > 0:
#         clustering = DBSCAN(eps=5, min_samples=10).fit(points)
#         labels = clustering.labels_
#         best_score = -1
#         best_box = None

#         for cluster_id in np.unique(labels):
#             if cluster_id == -1:
#                 continue
#             cluster_points = points[labels == cluster_id]
#             cluster_scores = [pred_mask[y, x] for y, x in cluster_points]
#             avg_score = np.mean(cluster_scores)

#             if avg_score > best_score:
#                 y_min, x_min = np.min(cluster_points, axis=0)
#                 y_max, x_max = np.max(cluster_points, axis=0)
#                 best_box = (x_min, y_min, x_max, y_max)
#                 best_score = avg_score

#         if best_box:
#             x_min, y_min, x_max, y_max = best_box
#             cv2.rectangle(overlay_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(overlay_with_box, f"box_score: {best_score:.2f}", (x_min, y_min - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     # === Save triplet visualization
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5))
#     axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
#     axes[0].set_title("Original Image")

#     axes[1].imshow(heatmap)
#     axes[1].set_title(f"Anomaly Heatmap\nScore: {scores[idx]:.4f}")

#     axes[2].imshow(cv2.cvtColor(overlay_with_box, cv2.COLOR_BGR2RGB))
#     axes[2].set_title("Best Scoring Cluster")

#     for ax in axes:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.savefig(output_path / f"bbox_clustered_tripletteee_{idx}.png")
#     plt.close()



# result_root = Path("/var/lib/containerd/battery_project/anode-defect-detector/results")
# anomaly_dir = result_root / "anomaly"
# normal_dir = result_root / "normal"
# anomaly_dir.mkdir(exist_ok=True)
# normal_dir.mkdir(exist_ok=True)

# # # === CSV setup
# # === CSV setup
# csv_path = result_root / "inference_resultsfinal.csv"
# with open(csv_path, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(["filename", "label", "score", "x_min", "y_min", "x_max", "y_max", "crop_width", "crop_height"])

#     def unnormalize(tensor):
#         mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
#         std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
#         return (tensor * std + mean).clamp(0, 1)

#     for idx in range(len(test_dataset)):
#         sample = test_dataset[idx]
#         image_tensor = sample["image"]
#         image_path = sample["image_path"]
#         filename = os.path.basename(image_path)

#         # ✅ Load original image size BEFORE resize
#         original_img = cv2.imread(str(image_path))
#         crop_height, crop_width = original_img.shape[:2] if original_img is not None else (256, 256)

#         image = unnormalize(image_tensor).permute(1, 2, 0).numpy()
#         image_bgr = (image * 255).astype(np.uint8)[:, :, ::-1]
#         pred_mask = masks[idx]
#         score = scores[idx]

#         label = "anomaly" if score > 2 else "normal"
#         save_dir = anomaly_dir if label == "anomaly" else normal_dir
#         overlay = image_bgr.copy()

#         x_min = y_min = x_max = y_max = None

#         if label == "anomaly":
#             norm_mask = cv2.normalize(pred_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#             threshold = np.percentile(norm_mask, 95)
#             points = np.column_stack(np.where(norm_mask > threshold))

#             if len(points) > 0:
#                 clustering = DBSCAN(eps=5, min_samples=10).fit(points)
#                 labels = clustering.labels_
#                 best_score = -1
#                 best_box = None

#                 for cluster_id in np.unique(labels):
#                     if cluster_id == -1:
#                         continue
#                     cluster_points = points[labels == cluster_id]
#                     cluster_scores = [pred_mask[y, x] for y, x in cluster_points]
#                     avg_score = np.mean(cluster_scores)

#                     if avg_score > best_score:
#                         y_min, x_min = np.min(cluster_points, axis=0)
#                         y_max, x_max = np.max(cluster_points, axis=0)
#                         best_box = (x_min, y_min, x_max, y_max)
#                         best_score = avg_score

#                 if best_box:
#                     x_min, y_min, x_max, y_max = best_box
#                     cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#         # === Save image
#         cv2.imwrite(str(save_dir / filename), overlay)

#         # === Write to CSV with original crop size
#         writer.writerow([
#             filename,
#             label,
#             round(score, 4),
#             x_min, y_min, x_max, y_max,
#             crop_width, crop_height
#         ])



# for idx in range(len(test_dataset)):
#     image_tensor = test_dataset[idx]["image"]
#     image = unnormalize(image_tensor).permute(1, 2, 0).numpy()
#     image_bgr = (image * 255).astype(np.uint8)[:, :, ::-1]
#     pred_mask = masks[idx]

#     # === Normalize heatmap
#     norm_mask = cv2.normalize(pred_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     heatmap = cv2.applyColorMap(norm_mask, cv2.COLORMAP_JET)

#     # === Save dual-panel visualization
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
#     axes[0].set_title("Original Image")

#     axes[1].imshow(heatmap)
#     axes[1].set_title(f"Anomaly Heatmap\nScore: {scores[idx]:.4f}")

#     for ax in axes:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.savefig(output_path / f"heatmap_only_{idx}.png")
#     plt.close()


import os
import cv2
import pandas as pd
from pathlib import Path
from unicodedata import normalize

# --- Config ---
ROI_OFFSETS = {
    "left": {"x_offset": 127, "y_offset": 448},
    "right": {"x_offset": 733, "y_offset": 448},
}
INDEX_MIN = 215
INDEX_MAX = 222

# --- Helper ---
def normalize_crop_filename(crop_name):
    if "_left_crop.png" in crop_name:
        base = crop_name.replace("_left_crop.png", ".tif")
    elif "_right_crop.png" in crop_name:
        base = crop_name.replace("_right_crop.png", ".tif")
    else:
        base = crop_name
    return normalize("NFKC", base)

# --- Main ---
def draw_defects_on_filtered_tifs(csv_path, input_root, output_root_ok, output_root_ng):
    output_root_ok = Path(output_root_ok)
    output_root_ng = Path(output_root_ng)
    output_root_ok.mkdir(parents=True, exist_ok=True)
    output_root_ng.mkdir(parents=True, exist_ok=True)

    print(f"📥 Reading defect CSV: {csv_path}")
    df = pd.read_csv(csv_path, header=None,
                     names=["filename", "label", "score", "x_min", "y_min", "x_max", "y_max"])
    df.dropna(subset=["x_min", "y_min", "x_max", "y_max"], inplace=True)

    if "anomaly" not in df["label"].unique():
        print("❌ No 'anomaly' label found in the CSV. Exiting early.")
        return

    defects_dict = {}
    label_dict = {}
    for _, row in df.iterrows():
        crop_name = row["filename"]
        label = row["label"]

        if "_left_crop" in crop_name:
            side = "left"
        elif "_right_crop" in crop_name:
            side = "right"
        else:
            continue

        original_name = normalize_crop_filename(crop_name)
        offset = ROI_OFFSETS[side]
        x_min = int(row["x_min"]) + offset["x_offset"]
        y_min = int(row["y_min"]) + offset["y_offset"]
        x_max = int(row["x_max"]) + offset["x_offset"]
        y_max = int(row["y_max"]) + offset["y_offset"]
        score = float(row["score"])

        if original_name not in defects_dict:
            defects_dict[original_name] = []
            label_dict[original_name] = []
        defects_dict[original_name].append((label, score, x_min, y_min, x_max, y_max))
        label_dict[original_name].append(label)

    input_root = Path(input_root)
    total_processed = 0
    total_anomaly = 0
    total_good = 0

    for root, _, files in os.walk(input_root):
        for file in files:
            full_path = Path(root) / file
            if (
                file.lower().endswith(".tif") and
                "負極" in file and
                "Z軸" in str(root) and
                not str(full_path).startswith(str(output_root_ok)) and
                not str(full_path).startswith(str(output_root_ng)) and
                any(f"_{i:04}" in file for i in range(INDEX_MIN, INDEX_MAX + 1))
            ):
                img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"⚠️ Could not read image: {full_path}")
                    continue

                save_folder = output_root_ok
                has_defect = file in defects_dict
                is_anomaly = False

                if has_defect:
                    for label, score, x_min, y_min, x_max, y_max in defects_dict[file]:
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    if any(l == "anomaly" for l in label_dict[file]):
                        save_folder = output_root_ng
                        is_anomaly = True

                save_path = save_folder / file.replace(".tif", ".png")
                cv2.imwrite(str(save_path), img)

                if is_anomaly:
                    total_anomaly += 1
                else:
                    total_good += 1

                total_processed += 1

    print("\n📊 Summary")
    print("-----------")
    print(f"📦 Total images processed: {total_processed}")
    print(f"✅ Good (OK) images:        {total_good}")
    print(f"❌ Anomaly (NG) images:     {total_anomaly}")

# --- Run ---
draw_defects_on_filtered_tifs(
    csv_path="/var/lib/containerd/battery_project/anode-defect-detector/results/inference_results.csv",
    input_root="/var/lib/containerd/battery_project/anode-defect-detector/data/raw/good_high_resolution",
    output_root_ok="/var/lib/containerd/battery_project/anode-defect-detector/final_results/OK",
    output_root_ng="/var/lib/containerd/battery_project/anode-defect-detector/final_results/NG"
)

# import os
# import cv2
# import pandas as pd
# from pathlib import Path
# from unicodedata import normalize
# from collections import defaultdict

# # === Accurate ROI config as used during cropping ===
# roi_config = [
#     {"x_min": 127, "x_max": 294, "y_min": 448, "y_max": 576, "name": "left"},
#     {"x_min": 733, "x_max": 900, "y_min": 448, "y_max": 576, "name": "right"},
# ]
# ROI = {
#     roi["name"]: {
#         "x_offset": roi["x_min"],
#         "y_offset": roi["y_min"],
#         "width": roi["x_max"] - roi["x_min"],
#         "height": roi["y_max"] - roi["y_min"],
#     }
#     for roi in roi_config
# }

# INDEX_MIN = 215
# INDEX_MAX = 222

# def extract_key(filename):
#     parts = filename.replace("_left_crop.png", "").replace("_right_crop.png", "").split("_")
#     return "_".join(parts[-2:])  # e.g., 9GP2105261Z001_0218

# def draw_anomalies(csv_path, input_root, output_root):
#     input_root = Path(input_root)
#     output_root = Path(output_root)
#     output_root.mkdir(parents=True, exist_ok=True)

#     print(f"📅 Reading CSV: {csv_path}")
#     df = pd.read_csv(csv_path)

#     # Ensure numeric columns are floats
#     numeric_cols = ["score", "x_min", "y_min", "x_max", "y_max", "crop_width", "crop_height"]
#     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
#     df.dropna(subset=numeric_cols, inplace=True)

#     defects_dict = defaultdict(lambda: {"left": [], "right": []})
#     csv_matches_by_key = defaultdict(list)

#     for _, row in df.iterrows():
#         label = row["label"]
#         if label != "anomaly":
#             continue

#         filename = normalize("NFKC", row["filename"])
#         side = "left" if "_left_crop" in filename else "right"
#         key = extract_key(filename)

#         crop_w, crop_h = row["crop_width"], row["crop_height"]
#         scale_x = crop_w / 256
#         scale_y = crop_h / 256

#         x_offset = ROI[side]["x_offset"]
#         y_offset = ROI[side]["y_offset"]

#         x_min = int(row["x_min"] * scale_x) + x_offset - 5
#         y_min = int(row["y_min"] * scale_y) + y_offset
#         x_max = int(row["x_max"] * scale_x) + x_offset + 5
#         y_max = int(row["y_max"] * scale_y) + y_offset

#         bbox = (float(row["score"]), x_min, y_min, x_max, y_max)
#         defects_dict[key][side].append(bbox)
#         csv_matches_by_key[key].append(filename)

#     print(f"🧠 Built defect dictionary with {len(defects_dict)} keys")

#     unmatched = []
#     matched = 0
#     matched_links = {}

#     for battery_folder in sorted(input_root.iterdir()):
#         if not battery_folder.is_dir():
#             continue

#         for tif_path in battery_folder.glob("*.tif"):
#             filename = tif_path.name
#             try:
#                 frame = int(filename.split("_")[-1].replace(".tif", ""))
#             except ValueError:
#                 print(f"⚠️ Skipping malformed filename: {filename}")
#                 continue

#             if not (INDEX_MIN <= frame <= INDEX_MAX):
#                 continue

#             battery_key = "_".join(filename.split("_")[-2:]).replace(".tif", "")
#             if battery_key not in defects_dict:
#                 unmatched.append(filename)
#                 continue

#             img = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
#             if img is None:
#                 print(f"❌ Cannot read image: {tif_path}")
#                 continue

#             found = False
#             for side in ["left", "right"]:
#                 for score, x_min, y_min, x_max, y_max in defects_dict[battery_key][side]:
#                     found = True
#                     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
#                     # cv2.putText(img, f"{side} {score:.2f}", (x_min, max(0, y_min - 5)),
#                     #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

#             if found:
#                 save_path = output_root / filename.replace(".tif", ".png")
#                 cv2.imwrite(str(save_path), img)
#                 matched += 1
#                 matched_links[filename] = csv_matches_by_key[battery_key]
#                 print(f"💾 Saved: {save_path}")

#     print("\n📊 Summary:")
#     print(f"✅ Annotated images: {matched}")
#     print(f"🚨 Unmatched images: {len(unmatched)}")
#     for u in unmatched:
#         print(f"   - {u}")

#     if matched_links:
#         print("\n🔗 Match details:")
#         for tif_file, csv_files in matched_links.items():
#             print(f"✅ Match: {tif_file}")
#             for crop_file in csv_files:
#                 print(f"    ↳ from CSV: {crop_file}")

# # --- Run ---
# draw_anomalies(
#     csv_path="/var/lib/containerd/battery_project/anode-defect-detector/results/inference_resultsfinal.csv",
#     input_root="/var/lib/containerd/battery_project/anode-defect-detector/data/raw/defects",
#     output_root="/var/lib/containerd/battery_project/anode-defect-detector/final_results/NG"
# )
