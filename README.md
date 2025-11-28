# OCT-CLINIC: Comparative Deep Learning for Retinal Disease Detection in OCT

This repository compares modern deep-learning architectures for **4-class OCT disease classification** (CNV, DME, DRUSEN, NORMAL) and provides a lightweight, deployment-friendly **SE-CNN** that matches or surpasses large pretrained backbones on the OCT2017 benchmark.

> **Headline**: Our OCT-specific **SE-CNN (~0.30M params)** reaches **~99.7% test accuracy** with **macro ROC-AUC = 1.000**, **low ECE (~0.04–0.06)**, and **~362 images/s** throughput. InceptionV3 baselines reach ~98–99% with ~22M parameters.

---

## Dataset Download

Use the **OCT2017** dataset (Kaggle mirror of UCSD/Mendeley; ~84k B-scans).  
Kaggle dataset: `paultimothymooney/kermany2018`

* If using **Kaggle Notebooks**, the dataset is usually available at:

```
/kaggle/input/kermany2018/OCT2017
```

* Otherwise download & extract to `data/OCT2017/`.

---

## Dataset Structure

```
OCT2017/
├── train/
│   ├── CNV/ ...
│   ├── DME/ ...
│   ├── DRUSEN/ ...
│   └── NORMAL/ ...
├── test/
│   ├── CNV/ ...
│   ├── DME/ ...
│   ├── DRUSEN/ ...
│   └── NORMAL/ ...
└── val/
    ├── CNV/ ...
    ├── DME/ ...
    ├── DRUSEN/ ...
    └── NORMAL/ ...
```

Official splits: **83,484 train / 32 val / 968 test**.  
A balanced test subset (542) is also common.

---

## Background

**Optical Coherence Tomography (OCT)** is essential for diagnosing CNV, DME and DRUSEN, but manual reads are time-consuming and variable.  
We build a **unified, reproducible pipeline** to compare off-the-shelf pretrained CNNs with a **small, OCT-tailored SE-CNN**, reporting not only accuracy but also **calibration, interpretability (Grad-CAM)** and **latency**.

---

## Problem Statement

Given an OCT B-scan, classify it into one of four disease categories:  
**CNV, DME, DRUSEN, NORMAL.**

We ask:  
**Can a compact, OCT-specific CNN match or beat large ImageNet-pretrained backbones while being faster, smaller, and well-calibrated?**

---

## Why is this important?

* Automates triage for clinical efficiency  
* Improves reliability via calibration (ECE) + saliency  
* Enables real-time & edge deployment with small models  

## Why is it challenging?

* OCT has grayscale textures and speckle noise  
* Dataset is imbalanced  
* Cross-device generalization is difficult  

---

## Installation

### Prerequisites

* Python **3.10+**  
* GPU (CUDA recommended)  
* 8–12GB VRAM for pretrained models  

### Setup

```bash
git clone https://github.com/nazish00/DeepLearningProject.git
cd DeepLearningProject

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── code/
│   ├── Final SE-CNN.ipynb                 # Lightweight OCT-specific model (best)
│   ├── InceptionV3 Model.ipynb            # InceptionV3 baseline
│   ├── Inception-ResNet-V2 Model.ipynb    # Inception-ResNet-V2 baseline
│   ├── Improved CNN + InceptionV3.ipynb   # Improved custom CNN + InceptionV3 experiments
│   └── Early Custom CNN (Prototype).ipynb # Prototype baseline
│
├── report/
│   └── Project_report7.pdf                # Full report with figures & tables
│
├── result/                                # Auto-created: checkpoints, logs, plots
├── demo.py
├── requirements.txt
└── README.md
```

---

## Usage

### Run via Notebooks (recommended)

1. Open a notebook from `code/` in Kaggle or Colab.  
2. Set dataset/output paths near the top:

```python
DATA_DIR = "/path/to/OCT2017"         # e.g., "/kaggle/input/kermany2018/OCT2017"
OUT_DIR  = "result"                   # artifacts will be saved here
```

3. Run all cells. Each notebook:

* builds loaders with identical augs/class-weights,  
* trains with early stopping & LR scheduling,  
* writes checkpoints + plots to `result/<model_name>/`.

---

### Optional: Convert to Scripts

Prefer scripts? Convert any notebook:

```bash
jupyter nbconvert --to script "code/Final SE-CNN.ipynb"
python "code/Final SE-CNN.py" --data-dir /path/to/OCT2017 --batch 32 --epochs 10
```

(Expose args like `--img-size`, `--lr`, `--augment` inside the exported script.)

---

## Training

Unified recipe (across models):

* **Augmentations**: small rotations/shifts/zoom + hflip; grayscale inputs for custom CNNs; gray→RGB replication for pretrained  
* **Loss/opt**: categorical cross-entropy with class weights; Adam + cosine decay (SE-CNN) or ReduceLROnPlateau (pretrained)  
* **Regularization**: dropout (0.10–0.30), weight decay (L2), label smoothing (ε=0.05)  
* **Batch/epochs**: batch 16–32; epochs 10–20  
* **Image sizes**: SE-CNN **256×256 (gray)**; InceptionV3 **299×299 (RGB)**  

---

## Evaluation

The notebooks compute:

* **Classification report**, **confusion matrix**  
* **ROC/PR curves**, **macro/micro AUC**  
* **Calibration** (reliability diagram + ECE)  
* **t-SNE embeddings** and **Grad-CAM saliency**  

Artifacts are saved to:

```
result/<model_name>/...
```

---

## Demo

A minimal `demo.py` is included for quick inference on single images or folders.

### Run on a single image:

```bash
python demo.py \
  --model result/se_cnn/best_model.keras \
  --image path/to/example_oct.png
```

### Run on a directory:

```bash
python demo.py \
  --model result/se_cnn/best_model.keras \
  --dir path/to/folder/
```

The script auto-infers input size/channels (gray vs RGB), preprocesses accordingly, and prints prediction probabilities (also saves a JSON file).

---

## Results

### Quick Highlights (SE-CNN)

* **Test**: ~99.7% accuracy, macro AUC = **1.000**, ECE ≈ **0.043**  
* **Efficiency**: ~0.30M parameters, ~362 images/s (batch 32)


### Final Comparison (OCT2017, official test)

| Model                | Input            | Pretrained | Params | Train Acc. | Test Acc.  | Macro F1 | Macro AUC |
|---------------------|------------------|------------|--------|------------|------------|----------|-----------|
| Early Custom CNN    | 256×256 (Gray)   | No         | ~0.26M | ≈98.5%     | **96.49%** | 0.964    | ~0.99     |
| Improved Custom CNN | 256×256 (Gray)   | No         | ~0.30M | ≈99.2%     | **96.50%** | —        | —         |
| InceptionV3         | 299×299 (RGB)    | Yes        | 21.8M  | ≈99.6%     | **98.90%** | 0.987    | >0.99     |
| EfficientNet-B3     | 300×300 (RGB)    | Yes        | 10.8M  | ≈99.5%     | **98.30%** | 0.982    | >0.99     |
| **SE-CNN (ours)**   | 256×256 (Gray)   | No         | ~0.30M | **≈99.9%** | **99.70%** | **0.999** | **1.000** |


We use the official OCT2017 split (83,484 / 32 / 968).  
The validation set is very small; results rely mainly on the held-out test plus calibration/interpretability.


---

## Credits

* **Dataset**: OCT2017 (UCSD/Mendeley; Kaggle mirror)  
* **Libraries**: TensorFlow/Keras, NumPy, Pandas, OpenCV, scikit-learn, Matplotlib/Seaborn  
* **Compute**: Kaggle/Colab GPUs (A100/T4/P100)  

---

## Appendix: requirements.txt

```txt
# Core DL stack (TensorFlow/Keras, as used in the notebooks)
tensorflow==2.15.0
keras==2.15.0

# Scientific Python
numpy>=1.24
pandas>=2.0
scipy>=1.10

# Imaging & viz
opencv-python>=4.8
pillow>=10.0
matplotlib>=3.7
seaborn>=0.12
scikit-image>=0.21

# ML utilities
scikit-learn>=1.3
tqdm>=4.66

# (Optional) Jupyter for running notebooks locally
# jupyterlab>=4.0
```

---

## Appendix: demo.py

This repository includes a standalone `demo.py` script for simple, local inference.  
It automatically:

* loads your saved Keras model (`.keras`, `.h5`, or SavedModel)  
* infers input size and channels  
* preprocesses grayscale → grayscale/RGB as needed  
* runs predictions on a single image or a folder  
* prints results and saves a JSON file  

### Usage

**Single image:**

```bash
python demo.py --model result/se_cnn/best_model.keras --image path/to/example_oct.png
```

**Directory:**

```bash
python demo.py --model result/se_cnn/best_model.keras --dir path/to/folder/
```

---

## References

1. See the full report with figures & tables: `report/Project_report7.pdf`

