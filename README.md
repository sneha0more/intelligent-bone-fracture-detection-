#  Intelligent Bone Fracture Detection  
### Deep Learning Pipeline · YOLOv8 · Faster R-CNN · COCO Preprocessing

This repository contains an end-to-end deep learning pipeline for **bone fracture detection** using X-ray images.  
It was developed as part of the NUS **BT5151 – Advanced Analytics & Machine Learning** module and demonstrates:

- Medical image preprocessing (resizing, normalisation)
- COCO annotation conversion + scaling
- Albumentations augmentation pipeline for X-rays
- Object detection model training with **YOLOv8 (Ultralytics)**
- Object detection model training with **Faster R-CNN (Detectron2)**
- Evaluation: mAP, precision–recall curves, confusion matrices
- End-to-end reproducible workflow for medical imaging tasks

> **Data Privacy Notice:**  
> Real medical X-ray datasets and trained model weights are **NOT included** in this repository due to privacy restrictions.  
> Only synthetic placeholder samples are provided in `examples/`.

---

## 1. Project Structure

intelligent-bone-fracture-detection/
│
├── README.md
├── requirements.txt
│
├── docs/
│ ├── report.pdf # Full project report
│ ├── presentation_deck.pdf # Presentation slides
│ ├── pr_curve_yolov8.png # YOLOv8 evaluation
│ ├── confusion_matrix_yolov8.png
│ ├── results_yolov8.csv

│
├── notebooks/
│ ├── 01_preprocessing.ipynb
│ ├── 02_yolov8_training.ipynb
│ └── 03_fasterrcnn.ipynb
│
├── src/
│ ├── preprocessing/
│ │ ├── dataset.py
│ │ ├── coco_to_yolo.py
│ │ └── preprocess.py
│ │
│ └── models/
│ ├── train_yolov8.py
│ ├── train_fasterrcnn_detectron2.py
│ └── inference_utils.py
│
├── data/
│ ├── yolov8_data.yaml # YOLO dataset config
│ ├── sample_raw/ # placeholder samples only
│ ├── sample_processed/
│ └── sample_annotations/
│
└── examples/
├── sample_predictions/
└── sample_annotations/


---

## 2. Problem Overview

Bone fracture diagnosis from X-ray images is often:

- slow  
- manually intensive  
- prone to human error  
- difficult to scale in emergency settings  

This project builds an **automated computer vision pipeline** to support radiologists by detecting fractures with modern deep learning techniques.

---

## 3. Model Architectures

### **YOLOv8 (Ultralytics)**
A fast, single-stage object detector suitable for radiology workflows where inference speed matters.

Outputs from training include:

- Precision–Recall curve
- Confusion matrix
- mAP scores  
(See `docs/`)

---

### **Faster R-CNN (Detectron2)**
A high-accuracy two-stage detector using:

- ResNet/ResNeXt backbone  
- Feature Pyramid Network (FPN)  
- Region Proposal Network (RPN)

Well suited for detecting subtle fractures in complex X-rays.

Evaluation curves and loss plots may be included in `docs/`.

---

## 4. Preprocessing Pipeline

Implemented in `src/preprocessing/` + `01_preprocessing.ipynb`.

Steps include:

1. **Load raw X-rays and COCO annotations**
2. **Resize images** (e.g., 512×512)
3. **Scale bounding boxes**
4. **Augment data** using Albumentations:
   - flips  
   - brightness/contrast  
   - small rotations  
   - Gaussian blur  
5. **Normalise** pixel intensities
6. Save images + labels in:
   - YOLO format  
   - COCO JSON format  

---

##  5. Training

### YOLOv8 Training

Update `data/yolov8_data.yaml` paths, then run:

```bash
python src/models/train_yolov8.py
This will:
load YOLOv8 model
train with your dataset
save results inside runs/fracture_yolov8/

Faster R-CNN Training (Detectron2)
Either run the notebook:

notebooks/03_fasterrcnn.ipynb


Or run the script:

python src/models/train_fasterrcnn_detectron2.py \
  --train-json path/to/train.json \
  --train-img path/to/train/images \
  --val-json path/to/val.json \
  --val-img path/to/val/images \
  --output outputs/fasterrcnn

## 6. Inference

Example for YOLOv8 inference:

python src/models/inference_utils.py \
  --model-type yolov8 \
  --weights path/to/best.pt \
  --image examples/sample_raw/example_xray.png \
  --output examples/sample_predictions/output.png

##  7. Evaluation

Evaluation metrics generated include:
mAP@0.5 and mAP@0.5:0.95
PR curves
Confusion matrices
Loss curves
These files are stored under docs/.

##  8. Data Privacy

This project does not include:
real medical X-ray images
raw patient annotations
trained weights
All examples are synthetic or placeholders.
Users must supply their own dataset following:
COCO format (for Detectron2)
YOLO format defined in data/yolov8_data.yaml

##  9. Documentation

See:
docs/report.pdf (full academic report)
docs/presentation_deck.pdf (slides)
These outline methodology, architecture, results, and limitations.

## 10. Acknowledgment

This project was originally created for NUS BT5151 – Advanced Analytics and Machine Learning (Group 22).
This repository represents my personal, portfolio-ready version with reorganised code, documentation, and reproducible training scripts.