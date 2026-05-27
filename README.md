# 🧠 NeuroScan AI — Brain Tumor Segmentation

A web-based clinical decision support system for automated 
brain tumor segmentation using deep learning (U-Net + Flask).

## 🎯 Results
| Patient | WT | TC | ET | Average |
|---------|----|----|-----|---------|
| Training (BraTS 2021) | - | - | - | **82.02%** |
| BraTS2021_00495 (Unseen) | 95.33% | 96.23% | 91.50% | **94.35%** |
| BraTS2021_00621 (Unseen) | 65.87% | 100% | 100% | **88.62%** |

## 🚀 Features
- ✅ Patient Registration & History Management
- ✅ Multi-modal MRI Upload (FLAIR, T1, T1CE, T2)
- ✅ Automated U-Net Segmentation
- ✅ Severity Classification (Low/Moderate/High Risk)
- ✅ Dice Score Calculation
- ✅ PDF Report Generation
- ✅ Dark Theme Clinical UI

## 🛠️ Tech Stack
- **Model:** PyTorch U-Net (31M parameters)
- **Backend:** Flask + SQLAlchemy
- **Database:** SQLite
- **Frontend:** Bootstrap 5 Dark Theme
- **PDF:** ReportLab
- **Dataset:** BraTS 2021 (1251 patients)

## ⚙️ Setup & Run
conda create -n brain_tumor python=3.12
conda activate brain_tumor
pip install -r requirements.txt
python app.py

## 👨‍💻 Developer
Raj Sahu — BTech Final Year, CSE (AI/ML)