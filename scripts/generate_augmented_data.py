import sys
from pathlib import Path
import yaml
import os
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.extract_defects import DefectExtractor
from src.data.augment_defects import DefectAugmentor
from src.utils.coco_utils import save_coco_json
from src.utils.split import split_and_save_coco_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

with open(PROJECT_ROOT / "src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def main():
    roi_config = cfg["roi_config"]
    for roi in roi_config:
        roi["circles_y"] = [513]
        
    augmentor = DefectAugmentor(roi_config=roi_config,config=cfg)
    augmentor.crop_and_save_rois(
        input_dir=cfg["good_images_dir"],
        output_dir=cfg["output_train_dir"]
    )
    extractor = DefectExtractor(
        annotations_root=cfg["annotations_root"],
        images_root=cfg["images_root"],
        output_dir=cfg["output_dir"],
        roi_config=roi_config
    )
    defects, imgs1, anns1, img_id, ann_id = extractor.extract()

    imgs2, anns2, _ = augmentor.augment_images_and_generate_coco(
        defects=defects,
        good_images_dir=cfg["good_images_dir"],
        dest_good_images_dir=cfg["converted_png_dir"],
        output_dir=cfg["output_dir"],
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
