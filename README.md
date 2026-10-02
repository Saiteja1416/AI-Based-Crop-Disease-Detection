# AI-Based Crop Disease Detection Using Deep Learning

## Project Overview

This project presents an AI-based crop disease classification and detection system using Computer Vision and Deep Learning. It combines MobileNetV2 for crop disease classification and YOLOv8n for disease detection and localization.

The system uses the PlantDoc dataset, which contains real-world plant leaf images representing different crop disease categories.

The objective is to explore deep learning techniques for identifying plant diseases and supporting agricultural disease monitoring.

## Objectives

- Develop a crop disease classification model using MobileNetV2.
- Implement disease detection and localization using YOLOv8n.
- Evaluate model performance using standard classification and object detection metrics.
- Analyze class-wise performance and model errors.
- Explore the potential of AI-assisted crop disease identification.

## Technologies Used

- Python
- TensorFlow and Keras
- PyTorch
- MobileNetV2
- YOLOv8
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Dataset

**Dataset:** PlantDoc Object Detection Dataset

- 29 disease and leaf categories
- Real-world plant images
- XML bounding-box annotations
- Object detection labels converted into YOLO format
- Bounding-box crops used for classification

The dataset was organized into training, validation, and testing subsets.

## Proposed System Architecture

```text
          Plant Leaf Image
                 |
                 v
        Image Preprocessing
                 |
          +------+------+
          |             |
          v             v
      MobileNetV2     YOLOv8n
          |             |
          v             v
   Disease Class     Disease Detection
   Probabilities     and Bounding Boxes
          |             |
          v             v
    Classification   Localization
          |             |
          +------+------+
                 |
                 v
          Evaluation Results
```

## Methodology

### 1. MobileNetV2 – Disease Classification

MobileNetV2 with ImageNet pretrained weights was used through transfer learning.

**Model configuration:**

- Input image size: 224 × 224
- Output classes: 29
- Global Average Pooling
- Dropout: 0.3
- Softmax output layer
- Fine-tuning with selected base-model layers trainable
- Adam optimizer

### 2. YOLOv8n – Disease Detection

YOLOv8n was trained to detect and localize plant disease categories using bounding-box annotations.

**Training configuration:**

- Model: YOLOv8n
- Image size: 640 × 640
- Maximum epochs: 50
- Batch size: 16
- GPU: NVIDIA Tesla T4
- Early stopping enabled

## Experimental Results

### MobileNetV2 Classification

| Metric | Result |
|---|---:|
| Test Accuracy | 53.78% |
| Weighted Precision | 54.17% |
| Weighted Recall | 53.78% |
| Weighted F1-Score | 53.01% |
| Macro F1-Score | 49.42% |
| Test Samples | 476 |
| Number of Classes | 29 |

### YOLOv8n Object Detection

| Metric | Result |
|---|---:|
| Test Precision | 58.7% |
| Test Recall | 60.0% |
| mAP@50 | 62.2% |
| mAP@50–95 | 49.0% |
| Test Images | 236 |
| Test Instances | 452 |

These values represent the results obtained in the project's test experiments.

## Model Comparison

| Model | Task | Test Performance |
|---|---|---|
| MobileNetV2 | Classification | 53.78% Accuracy |
| YOLOv8n | Object Detection | 62.2% mAP@50 |

Accuracy and mAP are different evaluation metrics and should not be interpreted as directly equivalent.

## Project Structure

```text
AI-Based-Crop-Disease-Detection/
│
├── notebooks/
│   └── Crop_Disease_Detection.ipynb
│
├── models/
│   ├── MobileNetV2_best_finetuned.keras
│   └── best.pt
│
├── results/
│   ├── graphs/
│   └── Final_Project_Results.xlsx
│
├── README.md
└── requirements.txt
```

*The structure above is the planned repository organization. Files will be added as they are uploaded.*

## Evaluation and Visualization

The project includes:

- Per-class F1-score analysis
- Confusion matrix
- Top 15 classification errors
- Training and validation accuracy curves
- Training and validation loss curves
- YOLOv8n test performance visualization
- Model comparison table

## Future Scope

- Improve classification performance through additional training and data augmentation.
- Address class imbalance and limited samples in rare categories.
- Explore more advanced deep learning architectures.
- Develop a web-based crop disease prediction interface.
- Investigate deployment for practical agricultural applications.

## Limitations

The models were evaluated on the PlantDoc dataset. Their reported performance does not establish accuracy under all field conditions. Additional testing with diverse field images and independent datasets would be needed before practical deployment.

## Author

**Sai Teja**

B.Tech – Electrical and Electronics Engineering

GitHub: [Saiteja1416](https://github.com/Saiteja1416)

## Acknowledgment

The PlantDoc dataset was used for model development and evaluation. Dataset usage should follow its applicable license and attribution requirements.
