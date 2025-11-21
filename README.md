# Cardiomegaly Detection Using Deep Learning

This project applies deep learning methods to classify chest X-ray images as **cardiomegaly** (enlarged heart) or **normal**. It uses the publicly available *Cardiomegaly Disease Prediction* dataset from Kaggle and implements an EfficientNet-B0 model trained and evaluated in Google Colab.

---

## Project Overview

Cardiomegaly is an important clinical indicator of cardiovascular disease. The goal of this project is to build an automated image-classification model that can assist in identifying this condition from X-ray scans.

This repo contains:
- The **Collab notebook** used for training and evaluation  
- Supporting documentation  
- AI usage notes  
- A written project report (coming soon)

---

## Model Summary
Model architecture:
- EfficientNet-B0 (ImageNet pretrained, frozen base)
- Batch Normalization
- Dense(256) + ReLU
- Dropout(0.2)
- Dense(2) Softmax output

Training details:
- Epochs: 10  
- Batch size: 16  
- Loss: Categorical Crossentropy  
- Optimizer: Adamax (lr=0.0005)

Performance:
- Test accuracy: **~67%**
- Balanced precision/recall across classes

---

## Dataset

Kaggle Dataset:  
**Cardiomegaly Disease Prediction using CNN**  
https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn

Dataset Structure:
- Train: 3550 images  
- Test: 1114 images  
- Labels: `true` (cardiomegaly), `false` (normal)

All images resized to **224 × 224** and normalized before training.

---

## How to Run

1. Open the notebook in Google Colab  
2. Upload and unzip the dataset  
3. Run preprocessing + training cells  
4. Evaluate on test set  
5. Modify architecture or hyperparameters as desired

---

## Requirements

- Python 3.10+
- tensorflow >= 2.10
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

(Colab already includes all dependencies.)

---

## Project Status

- [x] Data preprocessing  
- [x] EfficientNet-B0 training  
- [x] Evaluation (accuracy, confusion matrix, classification report)  
- [x] Draft report completed  
- [ ] Improve model performance  
- [ ] Add augmentation / fine-tuning  
- [ ] Upload final report  

---

## References

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling. *ICML*.  
- Cohen, J. P., Vivion, A., & Chaudhari, N. (2020). Medical Imaging for AI. *Nature Medicine*.  
- Kaggle Dataset (Rahimanshu, 2021)

---

## Acknowledgements

Special thanks to Kaggle dataset creators and academic researchers advancing medical imaging AI.



