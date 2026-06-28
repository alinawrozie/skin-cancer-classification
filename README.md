# Skin Lesion Classification for Cancer Diagnosis using Deep Learning

This repository contains the code and implementation of my MSc AI Dissertation, which compares the performance of classical Convolutional Neural Networks (CNNs), VGG16, EfficientNetV2, and Vision Transformers (ViT) in classifying skin cancer lesions. 

The classification is split into two primary tasks using the ISIC skin lesion dataset:
* **Task 1: Melanoma vs. Keratosis & Nevus** (Classifying malignant Melanoma against benign lesions).
* **Task 2: Keratosis vs. Melanoma & Nevus** (Classifying Seborrheic Keratosis against other lesions).

---

## 🚀 Key Features of the Project
This repository is a showcase-ready, refined implementation of deep learning architectures for skin cancer classification:
- **End-to-End ML Pipeline**: Preprocessing raw images, applying advanced augmentations, and serializing them into high-performance TFRecord datasets.
- **Multi-Model Comparison**: Evaluates and compares Custom CNN baseline, pre-trained VGG16, EfficientNetV2-B1, and Vision Transformer (ViT-B16).
- **5-Fold Cross-Validation**: Implements rigorous cross-validation to evaluate model performance across multiple metrics.
- **Modularized Codebase**: Organizes core preprocessing pipelines and Keras model definitions into a reusable `src/` library.
- **Rich Evaluations**: Incorporates multiple metrics tracking (Accuracy, Precision, Recall, F1, ROC-AUC) and visualizes predictions directly.

---

## 📊 Models & Architecture Compare
The project evaluates and compares four different deep learning architectures on the skin lesion classification tasks:

1. **Custom CNN**: A baseline convolutional model incorporating Dropout and Batch Normalization layers.
2. **VGG16**: A pre-trained VGG16 network fine-tuned with customized classification heads.
3. **EfficientNetV2-B1**: An optimized, resource-efficient model leveraging progressive learning and neural architecture search.
4. **Vision Transformer (ViT-B16)**: A state-of-the-art transformer architecture applying self-attention mechanisms to image patches.

---

## 📈 Evaluation Metrics & Performance
Models are evaluated using 5-Fold Cross-Validation across multiple metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 📊 Performance Comparison Table (Task 1: Melanoma vs. Seborrheic Keratosis & Nevus)

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (with Dropout)** | **0.803** | 0.618 | 0.508 | 0.469 | 0.608 |
| **VGG16 (Stratified KFold)** | 0.694 | 0.542 | 0.540 | 0.528 | 0.572 |
| **EfficientNetV2-B1 (Stratified KFold)** | 0.604 | 0.567 | 0.636 | 0.557 | 0.659 |
| **Vision Transformer (ViT-B16)** | 0.697 | **0.703** | **0.794** | **0.686** | **0.769** |

<p align="center">
  <img src="assets/roc_curves.png" alt="ROC Curves Comparison" width="600">
</p>

### 📊 Performance Comparison Table (Task 2: Seborrheic Keratosis vs. Melanoma & Nevus)

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (with Dropout)** | **0.861** | 0.733 | 0.616 | 0.644 | 0.776 |
| **VGG16 (Pre-trained)** | 0.625 | 0.594 | 0.683 | 0.555 | 0.742 |
| **EfficientNetV2-B1 (Stratified KFold)** | 0.684 | 0.601 | 0.688 | 0.579 | 0.735 |
| **Vision Transformer (ViT-B16)** | 0.775 | **0.757** | **0.875** | **0.764** | **0.844** |

---

## 🛠️ Installation & Setup

### 1. Clone the repository:
```bash
git clone https://github.com/alinawrozie/skin-cancer-classification.git
cd skin-cancer-classification
```

### 2. Dataset Placement:
Ensure your dataset files are placed in a sibling directory named `dataset` (or modify paths as needed):
```text
dissertation-901/
├── 22-24_CE901-CE911-CF981-SU_nawrozie_abdul_a/  <-- Repository Root
└── dataset/
    ├── task1/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── task2/
```

### 3. Generate TFRecords:
Open and run [DataExploration.ipynb](DataExploration.ipynb) to clean the images, perform augmentations, and serialize them into high-performance TFRecord files.

### 4. Train and Evaluate Models:
Navigate to the respective directories under Task 1 or Task 2 and execute the Jupyter Notebooks (e.g., [ViT-B16.ipynb](1%20Melanoma%20vs%20Keratosis%20and%20Nevus/ViT-B16/ViT-B16.ipynb)) to train and load cross-validation fold weights.
