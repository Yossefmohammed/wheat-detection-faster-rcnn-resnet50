# Wheat Detection with Faster R-CNN ResNet50

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Yossefmohammed/wheat-detection-faster-rcnn-resnet50)

This project demonstrates wheat detection using a Faster R-CNN model with a ResNet50 backbone, leveraging transfer learning for object detection on the Global Wheat Detection dataset.

## Project Overview
- **Goal:** Detect wheat heads in field images using deep learning.
- **Model:** Faster R-CNN with ResNet50 backbone (pretrained on COCO).
- **Dataset:** [Global Wheat Detection Challenge](https://www.kaggle.com/competitions/global-wheat-detection/data)

## Workflow
1. **Data Preparation:**
   - Load and preprocess bounding box annotations.
   - Split data into training and validation sets.
2. **Custom Dataset:**
   - Implement a PyTorch `Dataset` for image and bounding box loading.
3. **Model Setup:**
   - Load Faster R-CNN ResNet50, replace the head for binary classification (wheat vs. background).
4. **Training:**
   - Train for 5 epochs, monitoring loss.
5. **Evaluation:**
   - Run inference on validation images and visualize predictions.

## Key Results & Visualizations

### 1. Data Loading Example
![Load the model](https://github.com/Yossefmohammed/wheat-detection-faster-rcnn-resnet50/blob/main/Load%20th%20model.png)

### 2. Training Progress
![Training the model](https://github.com/Yossefmohammed/wheat-detection-faster-rcnn-resnet50/blob/main/Training%20the%20model.png)

### 3. Prediction Arrays
![Prediction arrays](https://github.com/Yossefmohammed/wheat-detection-faster-rcnn-resnet50/blob/main/prediction%20arrays.png)

### 4. Model Evaluation
![Evaluate the model](https://github.com/Yossefmohammed/wheat-detection-faster-rcnn-resnet50/blob/main/Evaluate%20the%20model.png)

### 5. Detection Results
![Detection wheats](https://github.com/Yossefmohammed/wheat-detection-faster-rcnn-resnet50/blob/main/Detection%20wheats.png)

## How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision pandas scikit-learn matplotlib pillow
   ```
2. Download the Global Wheat Detection dataset and adjust paths as needed in the notebook.
3. Run the notebook: `wheat-detection-faster-rcnn-resnet50.ipynb`

## References
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [PyTorch Detection Models](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- [Kaggle Competition](https://www.kaggle.com/competitions/global-wheat-detection)

## Contact Information
- **Email:** [ypssefmohammedahmed@gmail.com](mailto:ypssefmohammedahmed@gmail.com)
- **Phone:** +20 112 607 8938
