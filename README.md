<div align="center">

<img src="https://img.shields.io/badge/version-1.0.0-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/YOLOv8-Detection-EF3939?style=for-the-badge&logo=yolo&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" />

<br /><br />

```
███████╗ ██████╗ ██████╗ ████████╗██████╗  █████╗  ██████╗██╗  ██╗
██╔════╝██╔════╝██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝
█████╗  ██║     ██║   ██║   ██║   ██████╔╝███████║██║     █████╔╝ 
██╔══╝  ██║     ██║   ██║   ██║   ██╔══██╗██╔══██║██║     ██╔═██╗ 
███████╗╚██████╗╚██████╔╝   ██║   ██║  ██║██║  ██║╚██████╗██║  ██╗
╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                                                              AI ⬡
```

# EcoTrack AI

### *A Multi-Modal Framework for Microplastic Quantification and Chemical Characterization*

<br/>

> **Detect. Classify. Map. Act.**
> 
> Combining Computer Vision, Raman Spectroscopy, and Geospatial Intelligence  
> to automate the global fight against microplastic pollution.

<br/>

[🚀 Quick Start](#installation) · [📊 Model Performance](#performance-metrics) · [🏗 Architecture](#system-architecture) · [📁 Dataset](#dataset) · [📜 Citation](#citation)

</div>

---

## 📌 Table of Contents

- [Abstract](#abstract)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Training Artifacts](#training-artifacts)
- [Installation](#installation)
- [Usage](#usage)
- [Example Output](#example-output)
- [Methodology](#methodology)
- [Technologies](#technologies)
- [Applications](#applications)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)
- [Citation](#citation)

---

## Abstract

Microplastic pollution poses one of the most pervasive and undercharacterized threats to global ecosystems. Conventional laboratory workflows — manual microscopy and bench-top spectroscopy — are slow, operator-dependent, and impossible to scale for real-world monitoring programs.

**EcoTrack AI** addresses this gap with a reproducible, end-to-end multi-modal AI pipeline:

| Modality | Method | Output |
|---|---|---|
| 🔬 Microscopy Images | YOLOv8 Object Detection | Particle count & bounding boxes |
| 📈 Raman Spectroscopy | Random Forest Classifier | Polymer type & confidence |
| 🌍 Geo-Coordinates | DBSCAN Spatial Clustering | Pollution hotspot maps |
| 🖥️ Interactive UI | Streamlit Dashboard | Unified visual analytics |

The result is a **high-throughput, automated, and scientifically reproducible** microplastic monitoring framework deployable in laboratory and field settings.

---

## Key Features

```
✦  Automated Microplastic Detection    —  YOLOv8n fine-tuned on 577 annotated images
✦  Raman Polymer Classification        —  17,000+ spectra → Random Forest (multi-class)
✦  Geospatial Hotspot Mapping          —  DBSCAN clustering on GPS-tagged samples
✦  Multi-Modal Fusion Pipeline         —  Vision + Spectral + Spatial in one workflow
✦  Real-Time Streamlit Dashboard       —  Interactive upload, inference, and visualization
✦  CPU-Friendly Inference              —  No GPU required for deployment
✦  Research-Ready Artifacts            —  Full training logs, curves, and confusion matrices
✦  High-Throughput Processing          —  53.6ms average inference per image
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│   ┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐   │
│   │ Microscope    │   │  Raman Spectra  │   │  GPS Coordinates │   │
│   │   Images      │   │   (.jws / .csv) │   │   (lat, lon)     │   │
│   └──────┬────────┘   └────────┬────────┘   └────────┬─────────┘   │
└──────────┼─────────────────────┼──────────────────────┼────────────┘
           │                     │                      │
           ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│  IMAGE PIPELINE  │  │  SPECTRAL PIPELINE   │  │  SPATIAL ENGINE  │
│                  │  │                      │  │                  │
│  Letterbox Resize│  │  .jws Binary Decode  │  │  Coordinate      │
│  Mosaic Augment  │  │  Savitzky-Golay Filt │  │  Normalization   │
│  Flip / Contrast │  │  Baseline Correction │  │                  │
│        ↓         │  │  Min-Max Normalize   │  │        ↓         │
│   YOLOv8n Model  │  │  Feature Extraction  │  │    DBSCAN        │
│   (640×640 px)   │  │        ↓             │  │  Clustering      │
│        ↓         │  │  Random Forest       │  │                  │
│  Bounding Boxes  │  │  Polymer Classifier  │  │  Hotspot Regions │
│  Particle Count  │  │  Polymer Type + Conf │  │  Density Map     │
└────────┬─────────┘  └──────────┬───────────┘  └────────┬─────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │   STREAMLIT DASHBOARD  │
                    │                        │
                    │  • Detection View      │
                    │  • Spectra Plot        │
                    │  • Hotspot Map         │
                    │  • Summary Statistics  │
                    └────────────────────────┘
```

---

## Repository Structure

```
EcoTrack-AI/
│
├── 📁 data/
│   ├── raw/
│   │   ├── images/                  # Raw microscopy images
│   │   └── raman/                   # Raw .jws spectral files
│   │
│   └── processed/
│       ├── yolo_labels/             # YOLO annotation txts (x,y,w,h)
│       └── master_dataset.csv       # Fused feature dataset
│
├── 📁 models/
│   ├── best.pt                      # YOLOv8 best checkpoint
│   └── raman_classifier.pkl         # Trained Random Forest model
│
├── 📁 runs/
│   └── detect/train/
│       ├── results.png              # Loss & metric curves
│       ├── confusion_matrix.png     # Per-class confusion
│       ├── F1_curve.png             # F1 vs confidence
│       ├── PR_curve.png             # Precision-Recall curve
│       └── val_batch0_labels.jpg    # Validation batch preview
│
├── 📁 src/
│   ├── signal_proc.py               # Raman signal processing
│   ├── image_eng.py                 # CV preprocessing utilities
│   └── clustering.py                # DBSCAN spatial analysis
│
├── app.py                           # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

---

## Dataset

### 🔬 Computer Vision Dataset

| Property | Detail |
|---|---|
| Total Images | 577 microscopy images |
| Annotation Format | YOLO (normalized `x, y, w, h`) |
| Classes | Microplastic, Background |
| Split | Train / Validation |
| Image Source | Lab microscope captures |

### 📈 Raman Spectroscopy Dataset

| Property | Detail |
|---|---|
| Total Spectra | 17,000+ samples |
| Format | Binary `.jws` + CSV export |
| Labels | Polymer type (multi-class) |
| Preprocessing | Noise filtering, baseline correction, normalization |
| Source | Curated lab & environmental samples |

> ⚠️ **Class Imbalance Note:** PET polymer mixtures were supplemented with additional samples to address class distribution skew.

---

## Data Preprocessing

### Computer Vision Pipeline

```
Raw Image (any size)
       │
       ▼
Letterbox Resize ──────► 640 × 640 (aspect-ratio preserved)
       │
       ▼
Mosaic Augmentation ───► Combines 4 images for richer context
       │
       ▼
Random Horizontal Flip ► 50% probability
       │
       ▼
Brightness / Contrast ─► Stochastic jitter for robustness
       │
       ▼
YOLO Format Labels ────► Normalized (x_center, y_center, w, h)
```

### Raman Spectroscopy Pipeline

```
Binary .jws File
       │
       ▼
Byte Decoding ──────────► Raw intensity array extraction
       │
       ▼
Savitzky-Golay Filter ──► Noise suppression (smoothing)
       │
       ▼
Baseline Correction ────► Fluorescence background removal
       │
       ▼
Min-Max Normalization ──► Intensity range → [0, 1]
       │
       ▼
Feature Extraction ─────► Peak positions, widths, ratios
```

---

## Model Details

### 🤖 YOLOv8 Detection Model

| Parameter | Value |
|---|---|
| Architecture | YOLOv8n (nano) |
| Task | Object Detection |
| Image Size | 640 × 640 |
| Epochs | 20 |
| Batch Size | 16 |
| Classes | Microplastic / Background |
| Optimizer | AdamW |
| Inference Mode | CPU + GPU compatible |

### 🌲 Random Forest Classifier (Raman)

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Task | Multi-class Polymer Classification |
| Input Features | Spectral peaks, normalized intensities |
| Output | Polymer type + prediction confidence |
| Serialization | `raman_classifier.pkl` (scikit-learn) |

---

## Performance Metrics

<div align="center">

### YOLOv8 Detection — Validation Results

| Metric | Score |
|:---:|:---:|
| **mAP@0.50** | **0.801** |
| **Precision** | **0.775** |
| **Recall** | **0.737** |
| **F1 Score** | **0.755** |
| **Inference Speed** | **53.6 ms / image** |

</div>

> 📊 Full training curves, confusion matrix, and PR curves are saved under `runs/detect/train/`.

---

## Training Artifacts

All training evidence is saved and version-controlled for reproducibility:

```
runs/detect/train/
├── results.png              ← Loss curves (box, cls, dfl) + mAP over epochs
├── confusion_matrix.png     ← True vs predicted class heatmap
├── F1_curve.png             ← F1 score vs confidence threshold
├── PR_curve.png             ← Precision-Recall tradeoff
└── val_batch0_labels.jpg    ← Ground truth on validation batch
```

These artifacts verify **model convergence, generalization, and calibration** — a requirement for any research-grade submission.

---

## Installation

### Prerequisites

- Python 3.10+
- pip / conda
- (Optional) CUDA-enabled GPU for faster training

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/EcoTrack-AI.git
cd EcoTrack-AI

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 🖥️ Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### 🔍 Run Inference (YOLOv8)

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model("data/raw/images/sample.jpg")
results[0].show()
```

### 📈 Classify Raman Spectrum

```python
import pickle
import numpy as np

with open("models/raman_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

# features: array of spectral features (see src/signal_proc.py)
features = np.load("data/processed/example_features.npy")
polymer = clf.predict([features])
print(f"Detected Polymer: {polymer[0]}")
```

### 🗺️ Run Spatial Clustering

```python
from src.clustering import run_dbscan_hotspots

hotspots = run_dbscan_hotspots(
    coords_csv="data/processed/master_dataset.csv",
    eps=0.01,
    min_samples=5
)
hotspots.plot()
```

---

## Example Output

The EcoTrack AI pipeline produces:

```
📦 Per-Sample Output
├── 🟥 Bounding boxes over detected microplastic particles
├── 🔢 Particle count per image
├── 🧪 Polymer classification (e.g., PET, PP, PE) + confidence score
│
📊 Aggregated Output
├── 🗺️  Pollution density heatmap (DBSCAN clusters)
├── 📈  Spectral signature plots
└── 🖥️  Interactive Streamlit dashboard with all the above
```

---

## Methodology

```
Step 1 ─ DETECT         Upload microscopy image → YOLOv8 detects particles
           │
Step 2 ─ EXTRACT        Raman spectra collected at each detected site
           │
Step 3 ─ CLASSIFY       Random Forest maps spectra → polymer type
           │
Step 4 ─ CLUSTER        GPS coordinates → DBSCAN → pollution hotspots
           │
Step 5 ─ VISUALIZE      All outputs rendered in Streamlit dashboard
```

---

## Technologies

<div align="center">

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Computer Vision** | YOLOv8 (Ultralytics), OpenCV |
| **Machine Learning** | Scikit-learn (Random Forest), DBSCAN |
| **Signal Processing** | SciPy (Savitzky-Golay), NumPy |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Streamlit |
| **Serialization** | Pickle (`.pkl`), PyTorch (`.pt`) |

</div>

---

## Applications

| Domain | Use Case |
|---|---|
| 🌊 Marine Science | Ocean and coastal pollution monitoring |
| 💧 Water Safety | Drinking water quality screening |
| 🌱 Soil Science | Agricultural and terrestrial contamination |
| 🧪 Environmental Research | Long-term pollution trend analysis |
| 🔬 Laboratory Automation | High-throughput sample processing |
| 🛥️ Oceanography | Ship-based real-time sampling systems |

---

## Future Work

- [ ] **Nano-plastic Detection** — extend detection resolution below 1 µm
- [ ] **Real-Time Microscope Integration** — live camera inference pipeline
- [ ] **Edge AI Deployment** — ONNX export for Raspberry Pi / Jetson Nano
- [ ] **Transformer Spectral Models** — replace Random Forest with 1D-Transformer
- [ ] **Mobile Application** — iOS/Android field sampling companion
- [ ] **Cloud Inference API** — REST endpoint for remote lab integration
- [ ] **NOAA Dataset Fusion** — incorporate global marine density layers

---

## Author

<div align="center">

**Pratik Patil**  
AI/ML Research Project · March 2026

*For questions, collaborations, or dataset inquiries, please open an issue or submit a pull request.*

</div>

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full terms.

```
MIT License — Free to use, modify, and distribute with attribution.
```

---

## Citation

If you use EcoTrack AI in your research, please cite:

```bibtex
@software{patil2026ecotrack,
  author    = {Pratik Patil},
  title     = {EcoTrack AI: A Multi-Modal Framework for Microplastic 
               Quantification and Chemical Characterization},
  year      = {2026},
  month     = {March},
  url       = {https://github.com/yourusername/EcoTrack-AI},
  license   = {MIT}
}
```

---

<div align="center">

*Built with purpose. Deployed for the planet.*

⬡ **EcoTrack AI** · MIT License · 2026

</div>
"# Marine-Microplastic-Analysis-System-" 
