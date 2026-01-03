import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FractureDataset(Dataset):
    def __init__(self, annotation_path, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(annotation_path) as f:
            coco = json.load(f)

        self.images = coco['images']
        self.annotations = coco['annotations']
        self.categories = coco['categories']

        # Group annotations by image_id
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            self.image_id_to_annotations.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = os.path.join(self.image_dir, file_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.image_id_to_annotations.get(image_id, [])
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # convert to x_min, y_min, x_max, y_max
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
            image = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)


        return image, target


# Augmentation for training
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.03, contrast_limit=0.03, p=0.2),
        A.Rotate(limit=10, p=0.3),
        A.RandomScale(scale_limit=0.1, p=0.3),
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Basic transforms for validation (no augmentation)
def get_valid_transforms():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
