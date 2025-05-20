import os
import sys
import yaml
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from patchcore_inspection.src.patchcore.patchcore import PatchCore
from patchcore_inspection.src.patchcore.datasets.mvtec import MVTecDataset, DatasetSplit
from patchcore_inspection.src.patchcore.backbones import load
from patchcore_inspection.src.patchcore import sampler


class PatchCoreTrainer:
    def __init__(self, config_path: Path):
        self.project_root = PROJECT_ROOT
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.data_dir = (self.project_root / '..' / self.config["output_patchcore_dir"]).resolve()
        self.model_save_path = (self.project_root / self.config["output_dir_model_patchcore"]).resolve()
        self.model_save_path.mkdir(parents=True, exist_ok=True)

        self.classname = "patchcore_data"
        self.model = None

    def _load_config(self, config_path: Path) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _prepare_dataset(self):
        dataset = MVTecDataset(
            source=self.data_dir,
            classname=self.classname,
            resize=(168, 128),
            imagesize=(167, 128),
            split=DatasetSplit.TRAIN,
        )
        subset_indices = random.sample(range(len(dataset)), k=min(4000000, len(dataset)))
        subset = Subset(dataset, subset_indices)
        return DataLoader(subset, batch_size=128, shuffle=False)

    def _initialize_model(self):
        backbone = load("resnet50")
        backbone.name = "resnet50"

        self.model = PatchCore(device=self.device)
        self.model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            input_shape=(3, 167, 128),
            target_embed_dimension=1024,
            pretrain_embed_dimension=4096,
            faiss_on_gpu=True,
            device=self.device,
        )

    def train_and_save(self):
        self._initialize_model()
        dataloader = self._prepare_dataset()
        self.model.fit(dataloader)
        self.model.save_to_path(str(self.model_save_path))


if __name__ == "__main__":
    trainer = PatchCoreTrainer(config_path=PROJECT_ROOT / "config.yaml")
    trainer.train_and_save()
