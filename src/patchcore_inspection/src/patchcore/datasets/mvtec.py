import os
from enum import Enum
import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Lambda

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_CLASSNAMES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class MVTecDataset(Dataset):
    def __init__(
        self,
        source,
        classname=None,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TEST,
        train_val_split=1.0,
        flat=False,
        inference_mode=False  # ✅ NEW
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.flat = flat
        self.inference_mode = inference_mode
        self.classnames_to_use = [classname] if classname else _CLASSNAMES
        self.train_val_split = train_val_split
        self.imagesize = (3, imagesize, imagesize)

        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            Lambda(lambda x: x.float()),
            Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        if inference_mode:
            self.data_to_iterate = self._load_inference_data(source)
        else:
            self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if anomaly != "good" and mask_path:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.shape[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname if classname else "inference",
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def _load_inference_data(self, source):
        paths = []
        if os.path.isfile(source):
            paths = [source]
        elif os.path.isdir(source):
            for fname in sorted(os.listdir(source)):
                fpath = os.path.join(source, fname)
                if os.path.isfile(fpath) and fpath.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    paths.append(fpath)
        else:
            raise FileNotFoundError(f"❌ Inference path not found: {source}")

        return [[None, "good", p, None] for p in paths]  # Fake mask info

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        data_to_iterate = []

        for classname in self.classnames_to_use:
            if self.flat:
                classpath = os.path.join(self.source, classname)
            else:
                classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")

            if not os.path.exists(classpath):
                raise FileNotFoundError(f"❌ Folder not found: {classpath}")

            anomaly_types = sorted(os.listdir(classpath))
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                if not os.path.isdir(anomaly_path):
                    continue
                image_files = sorted(os.listdir(anomaly_path))
                image_paths = [os.path.join(anomaly_path, x) for x in image_files]
                imgpaths_per_class[classname][anomaly] = image_paths

                if self.train_val_split < 1.0:
                    n = len(image_paths)
                    idx = int(n * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = image_paths[:idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = image_paths[idx:]

                if anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    if os.path.exists(anomaly_mask_path):
                        mask_files = sorted(os.listdir(anomaly_mask_path))
                        if len(mask_files) == len(image_paths):
                            maskpaths_per_class[classname][anomaly] = [
                                os.path.join(anomaly_mask_path, x) for x in mask_files
                            ]
                        else:
                            print(f"⚠️ Mismatch mask/image: {classname}/{anomaly}")
                            maskpaths_per_class[classname][anomaly] = [None] * len(image_paths)
                    else:
                        maskpaths_per_class[classname][anomaly] = [None] * len(image_paths)
                else:
                    maskpaths_per_class[classname][anomaly] = [None] * len(image_paths)

                for i, img_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    mask_path = maskpaths_per_class[classname][anomaly][i]
                    data_to_iterate.append([classname, anomaly, img_path, mask_path])

        return imgpaths_per_class, data_to_iterate
