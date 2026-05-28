# 🧠 NeuroScan AI — Brain Tumor Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green?style=for-the-badge&logo=flask)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?style=for-the-badge&logo=nvidia)
![Dice Score](https://img.shields.io/badge/Dice%20Score-82.02%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-blue?style=for-the-badge)

**A complete AI-powered clinical decision support system for automated brain tumor segmentation**
Built using U-Net Deep Learning + Flask Web Application

[🏥 Features](#-features) • [📊 Results](#-results) • [⚙️ Setup](#️-setup--installation) • [🏗️ Architecture](#️-model-architecture) • [👥 Team](#-team)

</div>

---

## 🎯 About The Project

**NeuroScan AI** is a final year B.Tech project developed at **United University, Prayagraj**. It addresses a critical healthcare challenge — brain tumor diagnosis currently requires 30–60 minutes of manual analysis by specialized radiologists, with significant inter-observer variability. In India alone, over 40,000 new brain tumor cases are reported annually, and the shortage of neuroradiologists creates severe delays in diagnosis.

NeuroScan AI solves this by providing:
- ⚡ **Under 1 minute** automated analysis (vs 30–60 minutes manual)
- 🎯 **82.02% Dice Score** on BraTS 2021 training dataset
- 🏆 **94.35% Dice Score** on completely unseen patient data
- 🖥️ A complete **web application** for clinical workflow integration

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏥 Patient Registration | Register patients with name, age, gender, mobile |
| 📤 Multi-modal MRI Upload | Upload FLAIR, T1, T1CE, T2 in .nii / .nii.gz format |
| 🤖 AI Segmentation | Automated U-Net inference — Whole Tumor, Tumor Core, Enhancing Tumor |
| 📊 Dice Score Calculation | Real-time quantitative evaluation per tumor region |
| ⚠️ Severity Classification | Low / Moderate / High Risk based on tumor volume & ET ratio |
| 📄 PDF Report Generation | Downloadable clinical reports via ReportLab |
| 🗂️ Patient History | View, search, and delete patient scan history |
| 🌙 Dark Theme UI | Clinical-grade dark interface built with Bootstrap 5.3 |

---

## 📊 Results

### Training Performance (BraTS 2021 — 1251 Patients)

| Epoch | Avg Dice Score | Status |
|-------|---------------|--------|
| 1 | 73.02% | Initial learning |
| 2 | 78.61% | Rapid improvement |
| 3 | 79.03% | Continued improvement |
| 4 | 79.25% | Approaching peak |
| **5** | **82.02%** | ✅ **Best Model — Saved** |
| 6 | 74.31% | Overfitting detected — training stopped |

### Generalization Testing (Completely Unseen Patients)

| Patient | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) | Average |
|---------|-----------------|-----------------|----------------------|---------|
| BraTS2021_00499 | 97.80% | 97.50% | 92.50% | **95.93%** |
| BraTS2021_00495 | 95.33% | 96.23% | 91.50% | **94.35%** |
| BraTS2021_00621 | 65.87% | 100.00% | 100.00% | **88.62%** |

> ⚠️ Lower WT score on BraTS2021_00621 is due to very small tumor volume (0.492 cm³) where boundary precision is inherently more challenging — a known limitation of Dice metrics on small regions.

---

## 🏗️ Model Architecture

**U-Net with Skip Connections** — designed specifically for biomedical image segmentation.

```
Input: 4-channel MRI (FLAIR + T1 + T1CE + T2) — Shape: (4, 240, 240)
         │
    ┌────▼────┐
    │ Encoder │  64 → 128 → 256 → 512 channels (with MaxPool)
    └────┬────┘
         │ Skip Connections
    ┌────▼────┐
    │Bottleneck│  1024 channels
    └────┬────┘
         │ Skip Connections
    ┌────▼────┐
    │ Decoder │  512 → 256 → 128 → 64 channels (with TransposedConv)
    └────┬────┘
         │
Output: 3-channel masks (WT + TC + ET) — Shape: (3, 240, 240)
```

### Model Statistics

| Parameter | Value |
|-----------|-------|
| Total Parameters | 31,044,227 (~31 Million) |
| Model Size | ~118.42 MB |
| Input Shape | (4, 240, 240) |
| Output Shape | (3, 240, 240) |
| Loss Function | Dice Loss + Binary Cross-Entropy |
| Optimizer | Adam (lr = 1e-4) |
| GPU Used | NVIDIA GeForce GTX 1650 (4GB VRAM) |
| Inference Time | < 1 second |

---

## 🗂️ Dataset

**BraTS 2021** — Brain Tumor Segmentation Challenge Dataset

| Parameter | Details |
|-----------|---------|
| Total Patients | 1,251 cases |
| Data Format | NIfTI (.nii.gz) |
| Total Size | ~13 GB |
| MRI Modalities | 4 per patient (FLAIR, T1, T1CE, T2) |
| Image Dimensions | 240 × 240 × 155 voxels |
| Tumor Sub-regions | 3 (Whole Tumor, Tumor Core, Enhancing Tumor) |
| Data Split | 70% Train / 15% Val / 15% Test |

> 📥 Dataset available at [Kaggle — BraTS 2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

### MRI Modalities Explained

- **FLAIR** — Highlights tumor edema (Whole Tumor detection)
- **T1** — Shows normal brain anatomy (baseline reference)
- **T1CE** — Contrast-enhanced, identifies Enhancing Tumor (most aggressive region)
- **T2** — Shows fluid and edema extent (Tumor Core infiltration)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.5.1 |
| Web Framework | Flask 3.0.3 |
| Database | SQLite + Flask-SQLAlchemy 2.0.30 |
| Frontend | Bootstrap 5.3 + Jinja2 |
| MRI Processing | NiBabel 5.2.1 |
| Visualization | Matplotlib 3.8.4 |
| PDF Generation | ReportLab 4.2.0 |
| GPU Acceleration | CUDA 12.4 |
| Training Dataset | BraTS 2021 (1,251 patients) |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.12
- CUDA 12.4 compatible GPU (recommended)
- Anaconda / Miniconda

### Step 1 — Clone Repository
```bash
git clone https://github.com/Rajsahu760/NeuroScan-AI.git
cd NeuroScan-AI
```

### Step 2 — Create Conda Environment
```bash
conda create -n brain_tumor python=3.12
conda activate brain_tumor
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add Model Weights ⚠️
The trained model weights (`best_model.pth`, ~118 MB) are **not included** in this repository due to GitHub's file size limit.

Place the model file at:
```
models/best_model.pth
```

> 📩 Contact the team to request model weights.

### Step 5 — Run Application
```bash
python app.py
```

Open browser at: **http://localhost:5000**

---

## 📁 Project Structure

```
NeuroScan-AI/
│
├── app.py                          # Main Flask application & AI inference
├── config.py                       # Configuration (paths, database URI)
├── requirements.txt                # Python dependencies
├── base.html                       # Root HTML template
│
├── templates/                      # Jinja2 HTML templates
│   ├── base.html                   # Master layout with navbar
│   ├── index.html                  # Home dashboard
│   ├── register.html               # Patient registration
│   ├── upload.html                 # MRI upload page
│   ├── results.html                # Segmentation results
│   └── history.html                # Patient history
│
├── static/                         # Static assets
│   ├── css/                        # Stylesheets
│   ├── uploads/                    # Uploaded MRI files (gitignored)
│   └── results/                    # Generated segmentation images (gitignored)
│
├── models/                         # Model weights (gitignored — too large)
│   └── best_model.pth              # Place here after download
│
├── database/                       # SQLite database (gitignored — patient data)
│
├── 00_Dataset_Download_and_Extraction.ipynb   # Dataset setup
├── 01_BraTS_Data_Exploration.ipynb            # Data analysis & visualization
├── 02_Data_Preprocessing.ipynb               # Preprocessing pipeline
├── 03_Model_Architecture.ipynb               # U-Net implementation
├── 04_Training_Setup.ipynb                   # Model training
│
└── test_model.py                   # Standalone model testing script
```

---

## 🌐 Web Application Routes

| Route | Method | Function |
|-------|--------|----------|
| `/` | GET | Home dashboard with live statistics |
| `/register` | GET, POST | Patient registration form |
| `/upload/<id>` | GET, POST | MRI file upload + AI inference |
| `/results/<id>` | GET | Segmentation results & Dice scores |
| `/history` | GET | Patient history with search |
| `/download_report/<id>` | GET | PDF clinical report download |
| `/delete_patient/<id>` | POST | Delete patient and all records |

---

## ⚠️ Severity Classification

| Level | Label | Condition |
|-------|-------|-----------|
| Level 1 | 🟢 Low Risk | Tumor Volume < 5 cm³ OR ET Ratio < 0.20 |
| Level 2 | 🟡 Moderate Risk | Tumor Volume < 20 cm³ OR ET Ratio < 0.40 |
| Level 3 | 🔴 High Risk | Tumor Volume ≥ 20 cm³ AND ET Ratio ≥ 0.40 |

---

## 👥 Team

**Final Year B.Tech Project — Department of Computer Science & Engineering**
**United University, Rawatpur, Prayagraj, Uttar Pradesh — Session 2025–2026**

| Name | Enrollment No. |
|------|---------------|
| **Raj Sahu** | 22150050008 |
| **Himanshu Ranjan** | 22150050008 |
| **Abhay Tripathi** | 22150050002 |
| **Raj Giri** | 22150030003 |

### 🎓 Supervised By

**Ms. Kumkum Dubey**
Assistant Professor
Department of Computer Science & Engineering
United University, Prayagraj

---

## 🙏 Acknowledgements

We sincerely thank:
- **Ms. Kumkum Dubey** — Project guide, for valuable support and guidance throughout
- **Dr. Prashant Shukla** — HOD, Department of CSE, United University
- **Mr. Ashutosh Sir** — Lab staff, for technical support
- **BraTS 2021 Challenge** organizers for the publicly available benchmark dataset
- All faculty members of the Department of CSE, United University

---

## 📚 Key References

1. Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark," arXiv:2107.02314, 2021.
2. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015.
3. Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)," IEEE TMI, 2015.
4. Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," Nature Methods, 2021.

> Full reference list available in the [project report](./report/).

---

## 📝 Important Notes

| File / Folder | Included | Reason |
|---------------|----------|--------|
| `models/best_model.pth` | ❌ No | 118 MB — exceeds GitHub limit |
| `data/` | ❌ No | 13 GB — BraTS 2021 dataset |
| `database/*.db` | ❌ No | Patient data privacy |
| `static/uploads/` | ❌ No | Patient MRI files |
| `app.py`, `templates/`, `config.py` | ✅ Yes | Full source code |
| Jupyter Notebooks | ✅ Yes | Complete training pipeline |

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star!**

Made with ❤️ by Team NeuroScan AI | United University, Prayagraj | 2025–2026

</div>
