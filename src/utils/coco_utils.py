import os
import json

def save_coco_json(path, images, annotations):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "defect"}]
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
