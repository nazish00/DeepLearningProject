### **1. InceptionV3 Model.ipynb**  
This notebook implements **InceptionV3** using ImageNet pretrained weights.  
It includes:

- Train/val/test data loading  
- RGB conversion for grayscale OCT images  
- Two-stage training (freeze → fine-tune)  
- Evaluation on the test set  
- Confusion matrix, ROC, PR curves  

**Performance:**  
- Train Accuracy: ≈ **99.1%**  
- Test Accuracy: ≈ **99.1%**  

---

### **2. Inception-ResNet-V2 Model.ipynb**  
This notebook explores **Inception-ResNet-V2**, a deeper architecture that combines Inception filters with residual connections.  
Includes:

- Data preprocessing  
- Custom head for OCT classification  
- Fine-tuning with class weights  
- Full evaluation metrics  

**Performance:**  
- Train Accuracy: ≈ **90.6%**  
- Test Accuracy: **N/A** (test set evaluation not run in this notebook)  

---

### **3. Early Custom CNN (Prototype).ipynb**  
The first attempt at creating a custom CNN for OCT classification.  
Includes:

- Basic CNN architecture on 256×256 grayscale  
- Augmentation and class balancing  
- Training curves (loss & accuracy)  
- Test evaluation with confusion matrix  

**Performance:**  
- Train Accuracy: ≈ **83.9%**  
- Test Accuracy: ≈ **91.4%**  

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
- Improved CNN — Train: ≈ **84.4%**, Test: ≈ **96.5%**  
- InceptionV3 — Train: ≈ **98.5%**, Test: ≈ **98.1%**  

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
- Train Accuracy: ≈ **93.5%**  
- Test Accuracy: ≈ **99.7%**  
- Parameters: **0.30M**  
- AUC: **1.000**  
- ECE: **0.043**  
- Speed: **~362 images/sec**

---

## 📊 Final Comparison Table

| Model                     | Train Acc | Test Acc | Parameters | Pretrained | Notes |
|--------------------------|-----------|----------|------------|-----------|-------|
| Early Custom CNN         | ≈83.9%    | ≈91.4%   | 0.26M      | No        | Prototype baseline |
| Improved Custom CNN      | ≈84.4%    | ≈96.5%   | ~0.30M     | No        | Stronger baseline |
| InceptionV3              | ≈99.1%    | ≈99.1%   | 21.8M      | Yes       | Two-stage fine-tuning |
| Inception-ResNet-V2      | ≈90.6%    | N/A      | ~55M       | Yes       | Very deep pretrained model (no test eval) |
| **Final SE-CNN (Ours)**  | **≈93.5%**| **≈99.7%**| **0.30M** | **No**    | **Best performance, fastest, small size** |
