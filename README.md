# Bone Fracture Classification

Binary classification of bone fractures in X-ray images using YOLOv8 and OpenCV.

## Overview

This project trains a YOLOv8 classification model to detect whether an X-ray image contains a bone fracture or not. The trained model is then loaded through OpenCV's DNN module for inference, with several image preprocessing techniques tested to evaluate their effect on accuracy.

## Dataset

[Bone Fracture Multi-Region X-ray Data](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)

| Split | Images |
|-------|--------|
| Train | 9246   |
| Val   | 829    |
| Test  | 506    |

## Project Structure

```
bone-fracture-classification/
├── train.py          # YOLOv8 model training and ONNX export
├── preprocess.py     # OpenCV DNN inference with preprocessing comparison
├── best.onnx         # Trained model in ONNX format
└── README.md
```

## Results

| Preprocessing Method | Accuracy |
|----------------------|----------|
| Baseline (none)      | 98.6%    |
| Gaussian Blur        | 98.2%    |
| CLAHE + Gaussian     | 97.0%    |
| CLAHE                | 96.8%    |
| Canny Edge Detection | 47.0%    |

## Requirements

```
pip install ultralytics opencv-python numpy
```

## Usage

**Training:**
```bash
python train.py
```

**Inference with preprocessing comparison:**
```bash
python preprocess.py
```