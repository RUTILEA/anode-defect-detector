from src.inference.inference_patchcore import PatchCoreInference
from pathlib import Path
import sys

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PROJECT_ROOT))
    config_path = PROJECT_ROOT / "config.yaml"
    inference = PatchCoreInference(config_path=config_path)

    inference.run_patchcore_on_filtered_images(
        model_path=inference.model_save_path,
        base_folder=inference.data_dir,
        roi_config=inference.config.get("roi_config"),
        crop_output_base=(inference.output_path / "patchcore_crops").resolve(),
        final_overlay_base=inference.output_path,
        threshold=inference.config.get("patchcore_threshold"),
        resize_size=tuple(inference.config.get("resize_size"))
    )

    if inference.config.get("export_ai_csv_files"):
        inference.save_csv_results()
