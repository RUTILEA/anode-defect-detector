from pathlib import Path
from src.inference.inference_rf_detr import RFDETRInference 
import sys

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent 
    sys.path.append(str(PROJECT_ROOT))

    config_path = PROJECT_ROOT / "config.yaml"
    inference = RFDETRInference(config_path=config_path)

    dataset = PROJECT_ROOT / inference.config["for_prediction"]
    output_dir = PROJECT_ROOT / inference.config["output_inference_dir"] / "inference_rf_detr"

    inference.run_inference(
        dataset_dirs=[dataset],
        output_dir=output_dir,
        threshold=inference.config.get("rf_detr_threshold")
    )