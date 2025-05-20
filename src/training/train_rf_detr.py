import os
import sys
import yaml
import torch
import pynvml
from pathlib import Path

from rfdetr import RFDETRBase


class RFDETRTrainer:
    def __init__(self, config_path: Path):
        self.project_root = Path(__file__).resolve().parent.parent
        sys.path.append(str(self.project_root))

        self.config = self._load_config(config_path)
        self.device = self._select_device()
        self.dataset_dir = Path(self.config["output_coco_dir"]).resolve()
        self.output_dir = Path(self.config["output_dir_model_rf_detr"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = RFDETRBase(num_classes=1, device=self.device)

    def _load_config(self, config_path: Path) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _select_device(self) -> str:
        pynvml.nvmlInit()
        best_gpu = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

        if torch.cuda.is_available():
            return "cuda"
        else:
            print("⚠️ No GPU available, using CPU.")
            return "cpu"

    def train(self):
        self.model.train(
            dataset_dir=str(self.dataset_dir),
            epochs=15,
            batch_size=32,
            grad_accum_steps=1,
            lr=1e-3,
            output_dir=str(self.output_dir),
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_min_delta=0.001,
            num_workers=4,
            early_stopping_use_ema=True
        )


if __name__ == "__main__":
    trainer = RFDETRTrainer(config_path=Path(__file__).resolve().parent.parent / "config.yaml")
    trainer.train()
