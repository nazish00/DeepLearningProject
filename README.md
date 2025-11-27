# 🩺 OCT Retinal Disease Classification — Deep Learning Project

This repository contains a complete deep-learning pipeline for classifying retinal diseases using OCT (Optical Coherence Tomography) images.  
Multiple CNN architectures are implemented, compared, improved, and finally optimized into a lightweight **SE-CNN model** designed specifically for OCT images.

The work is based on the OCT2017 dataset and includes preprocessing, augmentation, training, evaluation, explainability, calibration, and performance benchmarking.

---

## 📁 Repository Contents

Below is a description of each notebook in this repository.

---

### **1. InceptionV3 Model.ipynb**  
This notebook implements **InceptionV3** using ImageNet pretrained weights.  
It includes:

- Train/val/test data loading  
- RGB conversion for grayscale OCT images  
- Two-stage training (freeze → fine-tune)  
- Evaluation on the test set  
- Confusion matrix, ROC, PR curves  

**Performance:**  
- Train Accuracy: **XX%**  
- Test Accuracy: **YY%**

---

### **2. Inception-ResNet-V2 Model.ipynb**  
This notebook explores **Inception-ResNet-V2**, a deeper architecture that combines Inception filters with residual connections.  
Includes:

- Data preprocessing  
- Custom head for OCT classification  
- Fine-tuning with class weights  
- Full evaluation metrics  

**Performance:**  
- Train Accuracy: **XX%**  
- Test Accuracy: **YY%**

---

### **3. Early Custom CNN (Prototype).ipynb**  
The first attempt at creating a custom CNN for OCT classification.  
Includes:

- Basic CNN architecture on 256×256 grayscale  
- Augmentation and class balancing  
- Training curves (loss & accuracy)  
- Test evaluation with confusion matrix  

**Performance:**  
- Train Accuracy: **XX%**  
- Test Accuracy: **YY%**

This notebook serves as the **baseline prototype** from which improved architectures were developed.

---

### **4. Improved CNN + InceptionV3.ipynb**  
This notebook contains:

#### 🔹 **Improved Custom CNN**
- More robust CNN design  
- Stronger regularization  
- Improved accuracy over the early prototype  

#### 🔹 **InceptionV3 (Refined Implementation)**
- Better preprocessing  
- Cleaner training loop  
- Better convergence & performance  

**Performance:**  
- Improved CNN — Train: **XX%**, Test: **YY%**  
- InceptionV3 — Train: **XX%**, Test: **YY%**

---

### **5. Final SE-CNN.ipynb (Main Model)**  
This is the final, optimized model and **main contribution** of the project.  
The SE-CNN uses:

- Depthwise-Separable Convs  
- Squeeze-and-Excitation blocks  
- Label smoothing  
- Cosine LR decay  
- Class weighting  
- Extensive augmentation  

Includes:  
- Confusion matrices (VAL & TEST)  
- ROC & PR curves (perfect AUC = 1.000)  
- Calibration (ECE)  
- Grad-CAM heatmaps  
- t-SNE embeddings  
- Throughput (images/sec)  
- Final comparison table  

**Performance (SE-CNN):**  
- Train Accuracy: **XX%**  
- Test Accuracy: **YY%**  
- Parameters: **0.30M**  
- AUC: **1.000**  
- ECE: **0.043**  
- Speed: **~362 images/sec**

---

## 📊 Final Comparison Table

| Model                     | Train Acc | Test Acc | Parameters | Pretrained | Notes |
|--------------------------|-----------|----------|------------|-----------|-------|
| Early Custom CNN         | XX%       | YY%      | 0.26M      | No        | Prototype baseline |
| Improved Custom CNN      | XX%       | YY%      | ~0.30M     | No        | Stronger baseline |
| InceptionV3              | XX%       | YY%      | 21.8M      | Yes       | Two-stage fine-tuning |
| Inception-ResNet-V2      | XX%       | YY%      | ~55M       | Yes       | Very deep pretrained model |
| **Final SE-CNN (Ours)**  | **XX%**   | **YY%**  | **0.30M**  | **No**    | **Best performance, fastest, small size** |

---

## 🧪 Dataset  
We use the **OCT2017** dataset containing 4 classes:

- CNV  
- DME  
- DRUSEN  
- NORMAL  

Images are 1-channel grayscale OCT B-scans.

---

## 📈 Features of This Project

- Custom CNN models  
- Pretrained InceptionV3 and Inception-ResNet-V2  
- Final optimized SE-CNN  
- Training/validation curves  
- Confusion matrices  
- ROC & PR curves  
- Calibration plots  
- Grad-CAM explainability  
- t-SNE feature visualization  
- High-speed inference testing  
- Clean comparison between all models  

---

## ▶️ How to Run

```bash
git clone [https://github.com/nazish00/DeepLearningProject.git](https://github.com/nazish00/DeepLearningProject.git)
cd DeepLearningProject
pip install -r requirements.txt
jupyter notebook
