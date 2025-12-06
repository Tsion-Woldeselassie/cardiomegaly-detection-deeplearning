# Cardiomegaly Detection Using Deep Learning

This project applies convolutional neural networks (CNNs) and transfer learning to classify chest X-ray images as **cardiomegaly present (true)** or **absent (false)**.  
The dataset consists of labeled chest X-ray images and the entire workflow included preprocessing, model training, evaluation, and interpretation.


## Project Overview

The goal of this project is to build a complete computer vision pipeline that:

- Loads and preprocesses chest X-ray images  
- Applies data augmentation to improve generalization  
- Trains a **baseline CNN**  
- Evaluates two transfer learning models: **MobileNetV2** and **EfficientNetB3**  
- Compares model performance using accuracy, loss curves, classification reports, and confusion matrices  

Cardiomegaly is subtle and difficult to detect visually, making this a realistic and challenging medical imaging problem.




## Dataset Description

- **Source:** Chest X-ray dataset containing cardiomegaly annotations  
- **Classes:**  
  - `true` → cardiomegaly present  
  - `false` → no cardiomegaly  
- **Image counts:**  
  - Training: **3,551 images**  
  - Validation: **887 images**  
  - Test: **1,114 images**  
- **Preprocessing steps:**  
  - Resizing each image to **224×224**  
  - Normalizing pixel values  
  - Data augmentation: rotation, translation, zoom, and contrast adjustments  

**Dataset challenges:**

- High variance in brightness and exposure  
- Medical artifacts (wires, implants, pacemakers)  
- Borderline cardiomegaly cases look extremely similar to normal cases  
- Anatomical differences across patients  



## Methods

###  Preprocessing

- `tf.data` pipeline with caching + prefetching  
- Normalization using `Rescaling(1./255)`  
- Data augmentation applied during training only:  
  - Small rotations  
  - Translations  
  - Zoom  
  - Contrast shifts  


### Models

### **1. Baseline CNN**
A simple convolutional neural network used as the starting point.  
This model included:

- Convolution + MaxPooling layers  
- Flatten → Dense layers  
- Dropout for regularization  
- Sigmoid output for binary classification  



### **2. MobileNetV2 (Transfer Learning)**

- Pretrained on ImageNet  
- Feature extraction with selective fine-tuning  
- Lightweight architecture suitable for fast training  
- Performance improved over the baseline but limited by dataset complexity  


### **3. EfficientNetB3 (Transfer Learning)**

- Pretrained on ImageNet  
- More expressive architecture with compound scaling  
- Higher recall for cardiomegaly, which is important for clinical screening  



## Results Summary

| Model | Recall | Notable Strengths |
|-------|--------------|------------------|
| **Baseline CNN** | 0.64 | Simple, stable, forms a solid reference |
| **MobileNetV2** | 0.65 | Lightweight, good generalization |
| **EfficientNetB3** | 0.88 (best overall)** | Strong recall on cardiomegaly cases |

### Additional Observations

- EfficientNetB3 captured heart-size differences better than MobileNetV2.  
- Baseline performance was surprisingly competitive because augmentation helped prevent overfitting.  
- Overall accuracy remained around ~0.63–0.64 due to dataset noise + medical complexity.


## Interpretation & Lessons Learned

The goal of this project was to build a model capable of detecting cardiomegaly from chest X-ray images. Across all experiments, recall—especially for the true (cardiomegaly) class—became the most important metric. Missing cardiomegaly cases can have serious clinical consequences, so I prioritized reducing false negatives over maximizing overall accuracy.

My baseline CNN was able to learn general patterns but could not capture enough spatial detail, resulting in unstable validation performance. Transfer learning improved results: MobileNetV2 achieved more stable behavior, while EfficientNetB3 delivered the strongest recall (0.88) and the lowest number of missed cardiomegaly cases. This confirmed that deeper pretrained models generalize better on subtle medical features.

Throughout the project, I made many attempts to improve accuracy. I experimented with multiple augmentation strategies (rotations, translations, zoom, contrast adjustments), tried different training pipelines, adjusted batch sizes, and even attempted to train EfficientNetB7 (more advanced model). That model crashed due to memory limits, and I had to restart with different approach. Despite all of these efforts, overall accuracy remained difficult to improve. This is largely due to the challenging nature of the dataset: images vary widely in brightness, contain medical devices, and include many borderline cases where cardiomegaly is visually subtle. Even advanced models struggle to separate these ambiguous examples.

Overall, the results show that deep learning can detect cardiomegaly with moderate performance, and recall can be improved by using stronger pretrained models. However, achieving high accuracy on this dataset would likely require more labeled data, clearer clinical annotations, or integrating segmentation methods that highlight the heart region directly. This project demonstrated the complexity of real medical imaging tasks and the limitations of working with difficult datasets, even when using state-of-the-art models.


## Future Improvements

If extending the project:

- Add **Grad-CAM** visualizations for interpretability  
- Use segmentation to measure **cardiothoracic ratio (CTR)** directly  
- Apply learning rate scheduling and hyperparameter tuning  
- Train on a larger dataset


## Reproducibility

This project includes:

- **Complete notebook** with all code and outputs  
- **README.md** describing data, methods, results, and interpretation  
- **ai_usage.md** documenting how AI assistance was used  


## AI Usage Disclosure

ChatGPT was used for:

- Code debugging (e.g., shape mismatches, augmentation placement)  



## References

- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*  
- Howard, A. et al. (2017). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*  
- TensorFlow Documentation – https://www.tensorflow.org/api_docs  
- Kaggle: Cardiomegaly X-ray Dataset (dataset source)

---



