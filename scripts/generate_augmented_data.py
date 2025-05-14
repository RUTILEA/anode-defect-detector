import sys
import os
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.extract_defects import extract_defects_from_annotations
from src.data.augment_defects import augment_images_and_generate_coco, split_and_save_coco_dataset
from src.utils.coco_utils import save_coco_json

with open(PROJECT_ROOT / "src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

roi_config = [
    {"x_min": 127, "x_max": 294, "y_min": 448, "y_max": 576, "circles_y": [513], "name": "left"},
    {"x_min": 733, "x_max": 900, "y_min": 448, "y_max": 576, "circles_y": [514], "name": "right"},
]

def main():
    defects, imgs1, anns1, img_id, ann_id = extract_defects_from_annotations(
        cfg["annotations_root"], cfg["images_root"], cfg["output_dir"], roi_config
    )

    imgs2, anns2, _ = augment_images_and_generate_coco(
        defects=defects,
        good_images_dir=cfg["good_images_dir"],
        dest_good_images_dir=cfg["converted_png_dir"],
        output_dir=cfg["output_dir"],
        roi_config=roi_config,
        img_id_start=img_id,
        ann_id_start=ann_id
    )

    all_imgs = imgs1 + imgs2
    all_anns = anns1 + anns2

    split_and_save_coco_dataset(
        coco_images=all_imgs,
        coco_annotations=all_anns,
        output_base_dir=cfg["output_coco_dir"],
        original_image_dir=cfg["output_dir"]
    )

    save_coco_json(
        os.path.join(cfg["output_dir"], "annotations.json"),
        all_imgs,
        all_anns
    )

if __name__ == "__main__":
    main()
