# Pneumonia Detection Using MobileNetV2 and Streamlit

A deep learning based web application for detecting pneumonia from chest X-ray images using transfer learning with MobileNetV2 and Streamlit deployment.

## 🚀 Live Demo

[Try the live demo here](https://pneumonia-detection-app-29.streamlit.app/)

---

# Project Overview

This project uses a Convolutional Neural Network (CNN) with Transfer Learning to classify chest X-ray images into:

- NORMAL
- PNEUMONIA

The application allows users to upload chest X-ray images through a Streamlit web interface and get real-time predictions with confidence scores.

The primary objective of this project was not only model training, but also building a complete end-to-end machine learning pipeline including:

- Data preprocessing
- Transfer learning
- Model evaluation
- Threshold tuning
- Deployment
- Inference pipeline
- Streamlit integration

---

# Dataset

Dataset Used:
Chest X-Ray Images (Pneumonia)

Dataset Structure:

```text
train/
    NORMAL/
    PNEUMONIA/

val/
    NORMAL/
    PNEUMONIA/

test/
    NORMAL/
    PNEUMONIA/