import os
import random
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from patchcore_inspection.src.patchcore.patchcore import PatchCore
from patchcore_inspection.src.patchcore.datasets.mvtec import MVTecDataset, DatasetSplit
from patchcore_inspection.src.patchcore.backbones import load
from patchcore_inspection.src.patchcore import sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

config_path = PROJECT_ROOT / "config.yaml"
print(f"📁 Loading config from: {config_path}")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

data_dir = (PROJECT_ROOT / '../' / cfg["output_patchcore_dir"]).resolve()
model_save_path = (PROJECT_ROOT / cfg["output_dir_model_patchcore"]).resolve()
model_save_path.mkdir(parents=True, exist_ok=True)

classname = "patchcore_data"
device = torch.device("cuda:0")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

backbone = load("resnet50")
backbone.name = "resnet50"

model = PatchCore(device=device)
model.load(
    backbone=backbone,
    layers_to_extract_from=["layer2", "layer3"],
    input_shape=(3, 167, 128),
    target_embed_dimension=1024,  # Reduced from 1024
    pretrain_embed_dimension=4096,  # Reduced from 2048
    faiss_on_gpu=True,
    # patchsize=3,
    # patchstride=1,
    # anomaly_score_num_nn=10,  # Fewer neighbors reduces noise
    # featuresampler=sampler.ApproximateGreedyCoresetSampler(percentage=0.1,device=device),  # Better coverage
    device=device,
)

train_dataset = MVTecDataset(
    source=data_dir,
    classname=classname,
    resize=(168, 128),
    imagesize=(167, 128),
    split=DatasetSplit.TRAIN,
)
subset_indices = random.sample(range(len(train_dataset)), k=min(10000, len(train_dataset)))
train_dataset = Subset(train_dataset, subset_indices)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

model.fit(train_loader)
model.save_to_path(str(model_save_path))

