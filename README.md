# 🫁 Pneumonia Detection Using MobileNetV2 and Streamlit

> A deep learning based web application for detecting pneumonia from chest X-ray images using Transfer Learning with MobileNetV2 and Streamlit deployment.

---

## 🚀 Live Demo

### **[✨ Try the Live Demo Here →](https://pneumonia-detection-app-29.streamlit.app/)**

---

## 📋 Project Overview

This project classifies chest X-ray images into:

- ✅ **NORMAL**
- ⚠️ **PNEUMONIA**

The system uses a pretrained **MobileNetV2** architecture as a feature extractor and performs binary classification using transfer learning.

### 🎯 Key Focus Areas

The project focuses not only on model training, but also on:

- 🔧 Data preprocessing & augmentation
- 🧠 Transfer learning optimization
- 📊 Model evaluation & threshold tuning
- 🌐 Cloud deployment & debugging
- ⚡ Real-time inference pipeline
- 🎨 Streamlit web interface

---

## 📂 Dataset

**Dataset Used:** Chest X-Ray Images (Pneumonia)

### Dataset Structure

```text
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset contains chest X-ray images for binary classification.

---

## 🛠️ Technologies Used

```
Python  |  TensorFlow/Keras  |  MobileNetV2  |  Streamlit
NumPy  |  PIL  |  Scikit-learn  |  Matplotlib
```

---

## 🧠 Transfer Learning Architecture

The project uses **MobileNetV2** pretrained CNN architecture.

### Model Architecture

```
┌─────────────────────────────────┐
│     MobileNetV2 (ImageNet)      │
│   include_top=False             │
└─────────────────────────────────┘
            ↓
┌─────────────────────────────────┐
│   GlobalAveragePooling2D()      │
└─────────────────────────────────┘
            ↓
┌─────────────────────────────────┐
│  Dense(128, activation='relu')  │
└─────────────────────────────────┘
            ↓
┌─────────────────────────────────┐
│      Dropout(0.5)               │
└─────────────────────────────────┘
            ↓
┌─────────────────────────────────┐
│  Dense(1, activation='sigmoid') │
│    (Binary Classification)      │
└─────────────────────────────────┘
```

---

## ✨ Why MobileNetV2?

| Aspect | Benefit |
|--------|---------|
| 🪶 **Lightweight** | Smaller model size |
| ⚡ **Fast Inference** | Real-time predictions |
| 🎯 **Medical Imaging** | Excellent performance on medical tasks |
| 🌐 **Cloud Ready** | Efficient deployment on cloud environments |
| 💰 **Low Cost** | Reduced computational requirements |

---

## 🔄 Data Preprocessing

Each image was:

- 📐 Resized to **150×150** pixels
- 🎨 Converted to **RGB** format
- ⚙️ Normalized to range **[0, 1]**
- 🔀 Converted into tensors before prediction

---

## 🎲 Data Augmentation

To improve generalization and reduce overfitting:

- 🔄 Rotation
- 🔍 Zoom
- ↔️ Horizontal Flip
- ⬌ Width & Height Shift

This helped the model learn more robust features from chest X-ray images.

---

## 🏋️ Training Strategy

### Phase 1 — Feature Extraction

- 🔒 MobileNetV2 base model **frozen**
- 📚 Only custom classifier layers trained
- 📉 Binary crossentropy loss
- ⚙️ Adam optimizer

### Phase 2 — Fine Tuning

- 🔄 Additional training epochs
- 🎚️ Threshold tuning applied
- 📈 Recall optimization for pneumonia class

---

## 🏥 Why Recall Matters in Medical AI

In medical diagnosis:

- ⚠️ **False negatives are dangerous**
- 🚨 **Missing pneumonia cases is more critical than false positives**

Therefore, the project specifically optimized for:

- 📊 **Pneumonia Recall** (97%)
- 🎯 **Pneumonia F1-score**

Instead of optimizing only overall accuracy.

---

## 📊 Model Evaluation

Evaluation metrics used:

- ✓ Accuracy
- 🎯 Precision
- 📈 Recall
- 🎪 F1-score
- 📋 Confusion Matrix

---

## 🏆 Final Performance

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|:---------:|:------:|:--------:|
| **NORMAL** | 0.94 | 0.76 | 0.84 |
| **PNEUMONIA** | 0.87 | **0.97** ⭐ | 0.92 |

### Overall Metrics

- **Accuracy:** ~89%
- **Macro F1-score:** ~88%
- **Weighted F1-score:** ~89%
- **Pneumonia Recall:** ~97% ⭐

### Confusion Matrix

```
                Predicted
              Normal  Pneumonia
Actual Normal    179        55
       Pneumonia  12       378  ← Only 12 missed cases!
```

**Interpretation:**
- ✅ 378 pneumonia images correctly detected
- ⚠️ Only 12 pneumonia cases missed
- 🎯 Model optimized toward **higher pneumonia recall**

---

## 🎚️ Threshold Tuning

Instead of using the default threshold:

```python
Default: 0.5  →  Optimized: 0.6
```

This improved prediction behavior and reduced unnecessary false positives.

---

## 🌐 Streamlit Application Features

The deployed application supports:

- 📤 Upload chest X-ray image
- ⚡ Real-time prediction
- 📊 Confidence score display
- 📈 Pneumonia probability display
- 🎨 Simple & intuitive UI
- ☁️ Cloud deployment ready

---

## 🚀 Deployment

The project was deployed using:

- ☁️ **Streamlit Community Cloud**
- 🔗 **GitHub integration**

### Challenges Solved

- ⚙️ TensorFlow compatibility issues
- 🐍 Python runtime mismatch
- 📦 Dependency conflicts
- 📁 Weight loading issues
- 🔄 Streamlit API compatibility

---

## 📁 Project Structure

```
xray_streamlit/
│
├── 📁 assets/
│
├── 📁 model/
│   └── mobilenet_weights.weights.h5
│
├── 📁 utils/
│   └── preprocess.py
│
├── 📄 app.py
├── 📄 requirements.txt
├── 📄 runtime.txt
├── 📄 README.md
└── 📓 pneumonia_detection_mobilenetv2.ipynb
```

---

## 🔧 Installation & Setup

### Step 1: Clone Repository

```bash
git clone <your-repository-link>
```

### Step 2: Move to Project Directory

```bash
cd xray_streamlit
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run Streamlit App

```bash
streamlit run app.py
```

---

## 📸 Sample Screenshots

### Home Page
```
assets/home_page.png
```

### Prediction Output
```
assets/prediction_output.png
```

---

## 💡 Important Engineering Learnings

This project involved solving real-world ML engineering problems:

- ✅ Environment reproducibility
- ✅ Cloud deployment debugging
- ✅ TensorFlow dependency issues
- ✅ Runtime compatibility
- ✅ Serialization mismatch
- ✅ Streamlit deployment failures

### Key Takeaway

```
┌─────────────────────────────────────────────────────────────┐
│  Training a model is only ONE part of ML engineering.       │
│  Reliable deployment and inference are EQUALLY important.   │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Limitations

### Purpose

This project is intended for:
- 📚 Educational purposes
- 🎓 Learning deep learning workflows
- 🧠 Understanding transfer learning
- 📦 Exploring deployment pipelines

### ❌ NOT a Clinically Validated Medical System

Limitations include:

- 📊 Limited dataset size
- 🔍 No external validation
- 🎯 No explainability module
- 🏥 No clinical calibration
- ⚖️ Potential dataset bias

---

## 🚀 Future Improvements

Possible enhancements:

- 🔥 GradCAM heatmaps
- 🧠 Explainable AI integration
- ⚡ FastAPI backend
- 🐳 Docker deployment
- 🎨 Better UI/UX
- 📊 Model monitoring
- 📦 Batch prediction support
- 📱 TFLite optimization

---

## 👨‍💻 Author

**Mohit Nirmal**

---

## 📝 License

This project is open-source and available for educational purposes.

---

<div align="center">

### ⭐ If you found this helpful, please give it a star!

[🚀 Try Live Demo](https://pneumonia-detection-app-29.streamlit.app/) | [📖 View Code](https://github.com/Mohit-25-tech/pneumonia-detection-streamlit)

</div>
