# 🐟 Fish Disease Detection Using Hybrid CNN and SVM Ensemble

This project presents a lightweight yet powerful deep learning pipeline for detecting fish diseases from images using a hybrid approach. It integrates deep feature extraction (MobileNetV2 and EfficientNetB0), dimensionality reduction (PCA), class balancing (SMOTE), and robust classification (Ensemble of SVMs).

## 📌 Overview

Diseases in aquaculture are a major threat to food security and economy. Manual detection is labor-intensive and error-prone. This project aims to automate fish disease classification with high accuracy using image-based machine learning.

## 🔍 Methodology

The proposed pipeline includes the following steps:

1. **Input**: Labeled fish disease image dataset
2. **Preprocessing**:
   - CLAHE (for local contrast enhancement)
   - Gaussian Blur (to reduce noise)
   - Resizing (to 224×224 pixels)
3. **Data Augmentation**:
   - Rotation
   - Horizontal & vertical flipping
   - Zooming
4. **Feature Extraction**:
   - MobileNetV2 (lightweight CNN)
   - EfficientNetB0 (balanced accuracy & efficiency)
5. **Feature Concatenation**: Combine extracted features from both CNNs
6. **Dimensionality Reduction**:
   - PCA to reduce redundancy and overfitting
7. **Class Balancing**:
   - SMOTE to generate synthetic samples for minority classes
8. **Classification**:
   - Ensemble of SVMs with soft voting

## 📁 Dataset

The dataset is publicly available:

- **Source**: [Mendeley Data](https://data.mendeley.com/datasets/x3fz2nfm4w/3)
- **Description**: Freshwater fish images, labeled with multiple disease types and healthy conditions.

## 📊 Results

- Achieved **>90% classification accuracy**
- Significantly improved generalization across multiple fish diseases
- Lightweight and suitable for real-time aquaculture deployment

## 🚀 Installation & Usage

### Requirements

- Python 3.8+
- TensorFlow / Keras
- Scikit-learn
- imbalanced-learn
- OpenCV
- NumPy, Pandas, Matplotlib

### Setup

```bash
git clone https://github.com/yourusername/fish-disease-detection
cd fish-disease-detection
pip install -r requirements.txt
