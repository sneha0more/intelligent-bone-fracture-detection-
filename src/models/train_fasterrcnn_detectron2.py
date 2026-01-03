import os
import argparse

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

# Initialize Detectron2 logger
setup_logger()


def register_datasets(train_json, train_img, val_json, val_img):
    """
    Registers COCO-format training and validation datasets.
    """
    print(f" Registering datasets...")

    register_coco_instances(
        "fracture_train", {}, train_json, train_img
    )
    register_coco_instances(
        "fracture_val", {}, val_json, val_img
    )

    print(f" Datasets registered successfully!")


def get_config(output_dir: str):
    """
    Loads a default Faster R-CNN config and applies custom settings.
    """
    cfg = get_cfg()

    from detectron2 import model_zoo
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    # Dataset registration names
    cfg.DATASETS.TRAIN = ("fracture_train",)
    cfg.DATASETS.TEST = ("fracture_val",)

    # Pretrained COCO weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.DATALOADER.NUM_WORKERS = 2

    # Solver settings
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000  # adjust according to dataset size
    cfg.SOLVER.STEPS = []       # no LR decay

    # Model settings
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # fracture

    # Output directory
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for bone fracture detection")

    parser.add_argument("--train-json", type=str, required=True, help="Path to COCO training JSON file")
    parser.add_argument("--train-img", type=str, required=True, help="Path to training images folder")

    parser.add_argument("--val-json", type=str, required=True, help="Path to COCO validation JSON file")
    parser.add_argument("--val-img", type=str, required=True, help="Path to validation images folder")

    parser.add_argument("--output", type=str, default="outputs/fasterrcnn", help="Output directory for results")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n Starting Faster R-CNN training...\n")

    # Register datasets
    register_datasets(
        train_json=args.train_json,
        train_img=args.train_img,
        val_json=args.val_json,
        val_img=args.val_img,
    )

    # Load configuration
    cfg = get_config(args.output)

    print(f" Output directory: {cfg.OUTPUT_DIR}")
    print(f" Training model with config: Faster R-CNN (ResNet + FPN)\n")

    # Start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n Training complete! Check the outputs folder for logs and model weights.\n")


if __name__ == "__main__":
    main()
