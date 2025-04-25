import os
import sys
import yaml
import torch
from pathlib import Path
import pynvml

from rfdetr import RFDETRBase

pynvml.nvmlInit()
best_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

if torch.cuda.is_available():
    print(f"🎯 Using GPU: {best_gpu} — {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU available, using CPU.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

config_path = PROJECT_ROOT / "config.yaml"
print(f"📁 Loading config from: {config_path}")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# === Resolve config paths as absolute ===
dataset_dir = Path(cfg["output_coco_dir"]).resolve()
output_dir = Path(cfg["output_dir_model_rf_detr"]).resolve()


print(f"📂 Dataset directory: {dataset_dir}")
print(f"💾 Output directory: {output_dir}")

# === Initialize and train model ===
model = RFDETRBase()
model.train(
    dataset_dir=str(dataset_dir),
    epochs=4,
    batch_size=16,
    grad_accum_steps=2,
    lr=1e-3,
    output_dir=str(output_dir),
    # early_stopping=True,
    # early_stopping_patience=5,
    # early_stopping_min_delta=0.01,
    # early_stopping_use_ema=True
)
