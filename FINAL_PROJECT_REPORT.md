# 🔬 EcoTrack AI: Multi-Modal Microplastic Analysis System
### *Comprehensive Project Report | 2026*

---

## 1. Introduction
Microplastic pollution has emerged as one of the most pervasive environmental crises of the 21st century. These plastic particles, typically smaller than 5 mm, infiltrate aquatic and terrestrial ecosystems, posing substantial risks to marine life and human health through ingestion and chemical seepage (Ref: Khanam et al., 2025). 

**EcoTrack AI** is a "Digital Detective" framework designed to automate the identification and characterization of these pollutants. By integrating Computer Vision (YOLOv8), Raman Spectroscopy (Random Forest), and Geospatial Intelligence (DBSCAN), the system replaces traditional, slow laboratory workflows with a high-throughput, scientifically reproducible AI pipeline.

---

## 2. Literature Review
The field of microplastic research has traditionally relied on manual microscopy and standalone spectroscopic techniques. While accurate, these methods are limited by labor-intensive procedures and low throughput (Ref: Blettler et al., 2018).

Key advancements analyzed for this report include:
- **Machine Learning Revolution:** Algorithms such as SVM, Random Forests, and CNNs have demonstrated success in classifying microplastics based on chemical signatures and visual characteristics.
- **Advanced Imaging:** Recent studies report that **Secondary-Ion Mass Spectrometry (SIMS)** imaging allows for rapid *in-situ* identification and spatial mapping of small MPs (1–50 µm) with a spatial resolution of 700 nm, without sample pretreatment (Ref: 2021 Research Data).
- **Spectral Optimization:** Current research (2024) indicates that incorporating the **C-H bond region (2500–3600 cm⁻¹)** of Raman spectra, rather than just the "fingerprint" region, significantly improves classification performance, especially for resolving complex polymers like ABS from PS.

---

## 3. Problem Statement
Manual laboratory workflows for microplastic analysis are:
1.  **Inefficient:** Sifting through thousands of microscope images is prone to human error and Fatigue.
2.  **Slow:** Chemical characterization of individual particles takes hours of lab time.
3.  **Fragmented:** Data from images, spectra, and GPS locations are rarely integrated into a single actionable report.

There is an urgent need for an automated system that can **See** (detect), **Identify** (classify chemistry), and **Map** (geospatial tracking) microplastics in near real-time.

---

## 4. Objectives
The primary goals of EcoTrack AI are:
- **Automation:** Replace manual counting with YOLOv8-driven object detection.
- **Chemical Accuracy:** Utilize Random Forest classifiers trained on a 17,000+ sample database for polymer identification.
- **Geospatial Mapping:** Track pollution hotspots globally using DBSCAN clustering.
- **Efficiency:** Achieve high-throughput processing with an average inference speed of <60ms per image.
- **Accessibility:** Provide a "Glassmorphism" Dark UI via Streamlit for researchers and policymakers.

---

## 5. Methodology

### 5.1 Data Collection and Preprocessing
The system utilizes two distinct datasets:
- **Visual Dataset:** 577 microscopy images annotated in YOLO format (normalized `x, y, w, h`). Preprocessing includes **Letterbox Resizing** (640x640) and **Mosaic Augmentation** to enhance model robustness.
- **Spectral Dataset:** 17,000+ Raman spectroscopy samples. Signal processing involves **Savitzky-Golay filtering** for noise suppression, **Baseline Correction** for fluorescence removal, and **Localized Normalization** for intensity range stabilization.

### 5.2 Exploratory Data Analysis (EDA)
EDA revealed a diverse distribution of polymers including Polyethylene (PE), Polypropylene (PP), Nylon, and Polystyrene (PS). To address class imbalance in PET mixtures, the dataset was supplemented with synthetic samples, ensuring the Random Forest model generalizes well across all common pollutants.

### 5.3 Model Selection
- **YOLOv8-nano:** Selected for its balance between accuracy and inference speed, allowing for real-time mobile/CPU-based deployment.
- **Random Forest:** Chosen for spectral classification due to its resilience against overfitting and interpretability of feature importance (spectral peaks).
- **DBSCAN:** Used for spatial clustering because it identifies density-based hotspots without requiring a pre-defined number of clusters (unlike K-Means).

### 5.4 Model Training
Model training focused on multi-modal fusion. The YOLOv8 model was fine-tuned over 20 epochs using the AdamW optimizer. The Raman classifier was trained using features extracted from both the **Fingerprint** and **C-H bond** regions (2500-3600 cm⁻¹) to resolve chemically similar polymers.

### 5.5 Model Evaluation
Performance was measured using standard AI benchmarks:
- **Detection (YOLOv8):** 80.1% mAP@0.5, 0.775 Precision, 0.737 Recall.
- **Classification (Random Forest):** F1-scores approaching 0.95 for dominant polymers after localized normalization.
- **Throughput:** ~53.6ms per sample.

---

## 6. Results and Discussion
The integrated pipeline successfully "fuses" visual detection with chemical characterization:
1.  **Detection:** YOLOv8 accurately identifies particle counts in messy environmental samples.
2.  **Characterization:** The "Virtual Robotic Probe" simulates chemical sampling, feeding spectral data into the RF model to provide a precise polymer verdict.
3.  **Visualization:** The Streamlit dashboard renders these results into unified density maps and polymer distributions.

**Discussion:** The use of localized spectral normalization (as suggested in the 2024 research paper) proved critical in maintaining high classification accuracy across varying environmental conditions (Ref: 2024 Findings).

---

## 7. Conclusion and Future Scope
EcoTrack AI represents a "Digital Twin" of a professional marine laboratory. It demonstrates how autonomous software can scale the global fight against plastic pollution by providing rapid, accurate, and multi-modal insights.

**Future Scope:**
- **Nano-plastic Extension:** Extending detection resolution below 1 µm.
- **Live Microscope Integration:** Real-time camera inference for ship-based sampling.
- **Edge Deployment:** Exporting models for Raspberry Pi/Jetson Nano for remote field use.
- **NOAA Dataset Fusion:** Incorporating global marine density layers for predictive modeling.

---

### **Authors & Acknowledgements**
*Prepared by Pratik Patil (AI/ML Research Project)*  
*Supported by Frontiers in Environmental Science & Recent Global Research Indices.*
