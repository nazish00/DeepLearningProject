### **1. InceptionV3**  
This model uses **InceptionV3** with ImageNet pretrained weights.  
It includes train/val/test loading, RGB conversion from grayscale OCT, two-stage training (freeze → fine-tune), and full evaluation with confusion matrix, ROC, and PR curves.  

**Performance:**  
- Train Accuracy: ≈ **99.1%**  
- Test Accuracy: ≈ **99.1%**  

---

### **2. Inception-ResNet-V2**  
This model explores **Inception-ResNet-V2**, combining Inception filters with residual connections.  
It includes data preprocessing, a custom classification head, fine-tuning with class weights, and evaluation metrics on the training/validation data.  

**Performance:**  
- Train Accuracy: ≈ **90.6%**  
- Test Accuracy: **90.6%** 

---

### **3. Early Custom CNN (Prototype)**  
This is the first custom CNN built for OCT classification on 256×256 grayscale images.  
It uses basic convolutional blocks, augmentation, class balancing, and evaluates performance using training curves and a confusion matrix.  

**Performance:**  
- Train Accuracy: ≈ **83.9%**  
- Test Accuracy: ≈ **91.4%**  

---

### **4. Improved Custom CNN + InceptionV3**  

#### 🔹 **Improved Custom CNN**  
A stronger baseline CNN with more robust design and heavier regularization, improving on the early prototype.  

**Performance (Improved CNN):**  
- Train Accuracy: ≈ **84.4%**  
- Test Accuracy: ≈ **96.5%**  

#### 🔹 **InceptionV3 (Refined Implementation)**  
A cleaner, better-tuned InceptionV3 pipeline with improved preprocessing and training loop.  

**Performance (InceptionV3 in this notebook):**  
- Train Accuracy: ≈ **98.5%**  
- Test Accuracy: ≈ **98.1%**  

---

### **5. Final SE-CNN (Main Model)**  
The final, optimized model and **main contribution** of the project.  
It uses depthwise-separable convolutions, Squeeze-and-Excitation blocks, label smoothing, cosine LR decay, class weighting, and extensive augmentation, plus detailed analysis (confusion matrices, ROC/PR, calibration, Grad-CAM, t-SNE, throughput).  

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
| InceptionV3              | ≈99.1%    | ≈99.1%   | 21.8M      | Yes       | Two-stage fine-tuning |
| Inception-ResNet-V2      | ≈90.6%    | 90.6%    | ~55M       | Yes       | Very deep pretrained model  |
| Early Custom CNN         | ≈83.9%    | ≈91.4%   | 0.26M      | No        | Prototype baseline |
| Improved Custom CNN      | ≈84.4%    | ≈96.5%   | ~0.30M     | No        | Stronger baseline |
| **Final SE-CNN (Ours)**  | **≈93.5%**| **≈99.7%**| **0.30M** | **No**    | **Best performance, fastest, small size** |
