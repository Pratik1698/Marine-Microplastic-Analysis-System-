import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
from PIL import Image
from ultralytics import YOLO
import reverse_geocoder as rg

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EcoTrack AI | Microplastic Detection & Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CUSTOM CSS — Professional Dark Theme
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* --- Import Google Font --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* --- Global --- */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* --- Main background --- */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1321 100%);
    }

    /* --- Sidebar polish --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #162033 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    /* --- Glass card mixin --- */
    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.08);
    }

    /* --- Metric cards --- */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.1);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    /* --- Section headers --- */
    .section-header {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.1) 0%, transparent 100%);
        border-left: 3px solid #38bdf8;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* --- Tech badge --- */
    .tech-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        margin: 0.15rem;
        font-size: 0.75rem;
        color: #7dd3fc;
        font-weight: 500;
    }

    /* --- Pipeline step --- */
    .pipeline-step {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.25s ease;
    }
    .pipeline-step:hover {
        border-color: rgba(56, 189, 248, 0.3);
        background: rgba(15, 23, 42, 0.6);
    }
    .step-number {
        display: inline-block;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        background: linear-gradient(135deg, #0ea5e9, #38bdf8);
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.8rem;
        color: #0f172a;
        margin-right: 0.6rem;
    }
    .step-title {
        font-weight: 600;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .step-desc {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.3rem;
        padding-left: 2.2rem;
    }

    /* --- Severity indicators --- */
    .severity-high {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 0.8rem;
        color: #fca5a5;
    }
    .severity-med {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 0.8rem;
        color: #fcd34d;
    }
    .severity-low {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 10px;
        padding: 0.8rem;
        color: #86efac;
    }

    /* --- Footer --- */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(100, 116, 139, 0.15);
        margin-top: 3rem;
    }

    /* --- Hide Streamlit branding --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- Dataframe styling --- */
    .stDataFrame {
        border: 1px solid rgba(56, 189, 248, 0.1) !important;
        border-radius: 10px !important;
    }

    /* --- Button polish --- */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.3) !important;
    }

    /* --- File uploader --- */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(56, 189, 248, 0.2) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(56, 189, 248, 0.1);
        color: #94a3b8;
        padding: 0.5rem 1.25rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.1) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_yolo_model():
    """Load the custom-trained YOLOv8 Microplastic detector."""
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading best.pt: {e}")
        return None

@st.cache_data
def load_and_clean_raman():
    """Load and prepare Raman Spectroscopy dataset."""
    df = pd.read_csv('final_processed_microplastics_17k.csv')
    polymers = ['PE', 'PS', 'PMMA', 'PTFE', 'NYLON']

    def clean_label(label):
        label_upper = str(label).upper()
        found = [p for p in polymers if p in label_upper]
        return "_".join(sorted(found)) if found else 'BACKGROUND'

    df['clean_category'] = df['category'].apply(clean_label)
    return df

@st.cache_resource
def train_raman_model(df):
    """Train Random Forest on Raman spectral data and return cross-val scores."""
    X = df.drop(columns=['category', 'source_file', 'clean_category'])
    y = df['clean_category']
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X, y)
    # Cross-validation for model evaluation
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=1)
    return model, X.columns, cv_scores

@st.cache_data
def generate_hotspot_map():
    """Combine global datasets and cluster hotspots using DBSCAN with country info."""
    try:
        df_adv = pd.read_csv('ADVENTURE_MICRO_FROM_SCIENTIST.csv')
        df_geo = pd.read_csv('GEOMARINE_MICRO.csv')
        df_sea = pd.read_csv('SEA_MICRO.csv')

        df_adv_clean = df_adv[df_adv['Total_Pieces_L'] > 0][['Latitude', 'Longitude']].copy()
        df_adv_clean['source'] = 'ADVENTURE'
        df_adv_clean['value'] = df_adv[df_adv['Total_Pieces_L'] > 0]['Total_Pieces_L'].values

        df_geo_clean = df_geo[df_geo['MP_conc__particles_cubic_metre_'] > 0][['Latitude', 'Longitude']].copy()
        df_geo_clean['source'] = 'GEOMARINE'
        df_geo_clean['value'] = df_geo[df_geo['MP_conc__particles_cubic_metre_'] > 0]['MP_conc__particles_cubic_metre_'].values

        df_sea_clean = df_sea[df_sea['Pieces_KM2'] > 0][['Latitude', 'Longitude']].copy()
        df_sea_clean['source'] = 'SEAMICROPLASTICS'
        df_sea_clean['value'] = df_sea[df_sea['Pieces_KM2'] > 0]['Pieces_KM2'].values

        df_all = pd.concat([df_adv_clean, df_geo_clean, df_sea_clean]).dropna().reset_index(drop=True)

        # Add country information via reverse geocoding
        coordinates = list(map(tuple, df_all[['Latitude', 'Longitude']].values))
        try:
            results = rg.search(coordinates)
            countries = [result['cc'] for result in results]
            df_all['Country'] = countries
        except Exception as e:
            df_all['Country'] = 'Unknown'
            st.warning(f"Geocoding error: {e}")

        # DBSCAN clustering
        coords = np.radians(df_all[['Latitude', 'Longitude']])
        db = DBSCAN(eps=500/6371.0, min_samples=20, algorithm='ball_tree', metric='haversine').fit(coords)
        df_all['Cluster'] = db.labels_

        return df_all
    except Exception as e:
        st.error(f"Mapping error: {e}")
        return pd.DataFrame(columns=['Latitude', 'Longitude', 'Country'])


# Pre-load shared assets
raman_df = load_and_clean_raman()
raman_model, raman_features, cv_scores = train_raman_model(raman_df)
yolo_model = load_yolo_model()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIDEBAR — Professional Navigation
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo & Title
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.25rem;">🔬</div>
        <h2 style="margin: 0; color: #f1f5f9; font-weight: 700; letter-spacing: -0.02em;">EcoTrack AI</h2>
        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.78rem; font-weight: 400;">
            Multi-Modal Microplastic Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "📊 Dashboard",
        "🔬 Visual Detection",
        "🧪 Spectral Analysis",
        "🌍 Geospatial Mapping"
    ], label_visibility="collapsed")

    st.markdown("---")

    # Project Info
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <p style="color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; font-weight: 600;">
            Project Information
        </p>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.3rem 0;">
            📂 <strong style="color: #cbd5e1;">Type:</strong> ML Mini Project
        </p>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.3rem 0;">
            👤 <strong style="color: #cbd5e1;">Author:</strong> Prajakta Salunkhe, Pratik Patil, Shreyash Kashid
        </p>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.3rem 0;">
            📅 <strong style="color: #cbd5e1;">Year:</strong> 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Tech Stack
    st.markdown("""
    <p style="color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; font-weight: 600;">
        Tech Stack
    </p>
    <div>
        <span class="tech-badge">YOLOv8</span>
        <span class="tech-badge">scikit-learn</span>
        <span class="tech-badge">OpenCV</span>
        <span class="tech-badge">DBSCAN</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Pandas</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System status
    status_color = "#22c55e" if yolo_model else "#ef4444"
    status_text = "All Systems Operational" if yolo_model else "Model Unavailable"
    st.markdown(f"""
    <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 0.6rem 0.8rem; text-align: center;">
        <span style="color: {status_color}; font-size: 0.8rem;">● {status_text}</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":

    # --- Hero Section ---
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.03em;">
            Marine Microplastic Analysis System
        </h1>
        <p style="color: #64748b; font-size: 1rem; margin: 0.5rem 0 0 0; max-width: 750px; line-height: 1.6;">
            A multi-modal deep learning pipeline for automated identification, chemical characterization,
            and geospatial tracking of marine microplastics using Computer Vision, Raman Spectroscopy,
            and Density-Based Clustering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Key Metrics ---
    st.markdown('<div class="section-header">📈 Model Performance Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("mAP@0.50", "80.1%")
    m2.metric("Precision", "77.5%")
    m3.metric("Recall", "73.7%")
    m4.metric("Raman CV Accuracy", f"{cv_scores.mean()*100:.1f}%")
    m5.metric("Spectral Samples", f"{len(raman_df):,}")

    # --- Methodology Pipeline ---
    st.markdown('<div class="section-header">🔄 End-to-End Pipeline</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    steps = [
        ("1", "Image Acquisition", "Microscope images captured at varying magnifications"),
        ("2", "Object Detection", "YOLOv8n identifies and localizes microplastic particles"),
        ("3", "Chemical Analysis", "Random Forest classifies polymer type from Raman spectra"),
        ("4", "Geospatial Mapping", "DBSCAN clusters GPS coordinates into pollution hotspots"),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="pipeline-step">
                <span class="step-number">{num}</span>
                <span class="step-title">{title}</span>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Two-column: Model Details + Class Distribution ---
    st.markdown('<div class="section-header">🧠 Model Architecture</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #e2e8f0; margin-top: 0;">YOLOv8n — Object Detection</h4>
            <table style="width: 100%; color: #94a3b8; font-size: 0.85rem;">
                <tr><td style="padding: 0.3rem 0;">Architecture</td><td style="text-align: right; color: #e2e8f0;">YOLOv8 Nano</td></tr>
                <tr><td style="padding: 0.3rem 0;">Input Resolution</td><td style="text-align: right; color: #e2e8f0;">640 × 640 px</td></tr>
                <tr><td style="padding: 0.3rem 0;">Training Images</td><td style="text-align: right; color: #e2e8f0;">577</td></tr>
                <tr><td style="padding: 0.3rem 0;">Epochs</td><td style="text-align: right; color: #e2e8f0;">20</td></tr>
                <tr><td style="padding: 0.3rem 0;">Optimizer</td><td style="text-align: right; color: #e2e8f0;">AdamW</td></tr>
                <tr><td style="padding: 0.3rem 0;">Inference</td><td style="text-align: right; color: #e2e8f0;">53.6 ms / image</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #e2e8f0; margin-top: 0;">Random Forest — Spectral Classifier</h4>
            <table style="width: 100%; color: #94a3b8; font-size: 0.85rem;">
                <tr><td style="padding: 0.3rem 0;">Algorithm</td><td style="text-align: right; color: #e2e8f0;">Random Forest</td></tr>
                <tr><td style="padding: 0.3rem 0;">Estimators</td><td style="text-align: right; color: #e2e8f0;">50 trees</td></tr>
                <tr><td style="padding: 0.3rem 0;">Input Features</td><td style="text-align: right; color: #e2e8f0;">Spectral intensities</td></tr>
                <tr><td style="padding: 0.3rem 0;">Classes</td><td style="text-align: right; color: #e2e8f0;">PE, PS, PMMA, PTFE, NYLON</td></tr>
                <tr><td style="padding: 0.3rem 0;">Cross-Val (3-fold)</td><td style="text-align: right; color: #e2e8f0;">""" + f"{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%" + """</td></tr>
                <tr><td style="padding: 0.3rem 0;">Training Samples</td><td style="text-align: right; color: #e2e8f0;">""" + f"{len(raman_df):,}" + """</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # --- Class Distribution Chart ---
    st.markdown('<div class="section-header">📊 Dataset Class Distribution</div>', unsafe_allow_html=True)

    class_counts = raman_df['clean_category'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#0ea5e9', '#22d3ee', '#38bdf8', '#06b6d4', '#67e8f9', '#a5f3fc', '#7dd3fc']
    bars = ax.barh(class_counts.index, class_counts.values, color=colors[:len(class_counts)], edgecolor='none', height=0.6)
    ax.set_facecolor('#0a0e1a')
    fig.set_facecolor('#0a0e1a')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.set_xlabel('Sample Count', color='#94a3b8', fontsize=10)
    ax.xaxis.label.set_color('#94a3b8')
    for bar, val in zip(bars, class_counts.values):
        ax.text(val + max(class_counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', color='#94a3b8', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PAGE: VISUAL DETECTION (Computer Vision)
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Visual Detection":

    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2rem; font-weight: 700; margin: 0;">
            Automated Particle Detection
        </h1>
        <p style="color: #64748b; font-size: 0.9rem; margin: 0.3rem 0 0 0;">
            Upload a microscope image for real-time microplastic detection using the trained YOLOv8 model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model info bar
    st.markdown("""
    <div class="glass-card" style="padding: 0.8rem 1.2rem;">
        <span style="color: #94a3b8; font-size: 0.82rem;">
            <strong style="color: #38bdf8;">Model:</strong> YOLOv8n &nbsp;|&nbsp;
            <strong style="color: #38bdf8;">Confidence Threshold:</strong> 0.25 &nbsp;|&nbsp;
            <strong style="color: #38bdf8;">Input Size:</strong> 640×640 &nbsp;|&nbsp;
            <strong style="color: #38bdf8;">Inference:</strong> ~53.6ms
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload microscope sample image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        if yolo_model:
            with st.spinner("Running YOLOv8 inference..."):
                results = yolo_model.predict(image_rgb, conf=0.25)
                res_plotted = results[0].plot()
                count = len(results[0].boxes)
                confidences = results[0].boxes.conf.cpu().numpy() if count > 0 else np.array([])

                # Results header
                if count > 20:
                    severity_class = "severity-high"
                    severity_msg = f"🚨 HIGH DENSITY — {count} microplastic particles detected"
                elif count > 5:
                    severity_class = "severity-med"
                    severity_msg = f"⚠️ MODERATE PRESENCE — {count} microplastic particles detected"
                else:
                    severity_class = "severity-low"
                    severity_msg = f"✅ LOW PRESENCE — {count} microplastic particles detected"

                st.markdown(f'<div class="{severity_class}">{severity_msg}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # Image + Stats
                img_col, stats_col = st.columns([3, 1])

                with img_col:
                    tab1, tab2 = st.tabs(["🎯 Detection Overlay", "🖼️ Original Image"])
                    with tab1:
                        st.image(res_plotted, caption="YOLOv8 Detection Results", use_container_width=True)
                    with tab2:
                        st.image(image_rgb, caption="Original Input Image", use_container_width=True)

                with stats_col:
                    st.metric("Particles Found", count)
                    if len(confidences) > 0:
                        st.metric("Avg Confidence", f"{confidences.mean():.1%}")
                        st.metric("Max Confidence", f"{confidences.max():.1%}")
                        st.metric("Min Confidence", f"{confidences.min():.1%}")

                    # 🔗 NEW: MULTI-MODAL ANALYSIS INTEGRATION
                    st.markdown("---")
                    st.markdown("""
                    <p style="color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.8rem; font-weight: 600;">
                        Integrated Analysis
                    </p>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🧪 Run Multi-Modal Analysis", type="secondary", use_container_width=True):
                        st.session_state.run_analysis = True
                    
                    # Download report
                    report_lines = [
                        "=" * 45,
                        "  ECOTRACK AI — DETECTION REPORT",
                        "=" * 45,
                        f"  Particles Detected : {count}",
                        f"  Model              : YOLOv8n (Custom)",
                        f"  Confidence Thresh  : 0.25",
                        f"  Input Resolution   : {opencv_image.shape[1]}x{opencv_image.shape[0]}",
                    ]
                    if len(confidences) > 0:
                        report_lines += [
                            f"  Avg Confidence     : {confidences.mean():.4f}",
                            f"  Max Confidence     : {confidences.max():.4f}",
                            f"  Min Confidence     : {confidences.min():.4f}",
                        ]
                    report_lines += ["=" * 45]
                    report = "\n".join(report_lines)
                    st.download_button("📄 Download Basic Report", report, file_name="detection_report.txt", use_container_width=True)

                # --- 🔗 MULTI-MODAL RESULTS SECTION ---
                if st.session_state.get('run_analysis', False) and count > 0:
                    st.markdown('<div class="section-header">🔬 Automated Multi-Modal Results</div>', unsafe_allow_html=True)
                    
                    with st.spinner("Simulating robot spectral capture for each particle..."):
                        # Sample spectra for each particle
                        analysis_samples = raman_df.sample(n=min(count, 50)) # Cap at 50 for performance
                        analysis_features = analysis_samples[raman_features]
                        predictions = raman_model.predict(analysis_features)
                        
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            # Plot the first 3 particles' spectra
                            fig, ax = plt.subplots(figsize=(10, 5))
                            colors_p = ['#0ea5e9', '#22d3ee', '#38bdf8']
                            for i in range(min(len(predictions), 3)):
                                y_vals = analysis_features.values[i][:300]
                                ax.plot(y_vals, label=f"Particle {i+1}: {predictions[i]}", color=colors_p[i], lw=1.5, alpha=0.8)
                                ax.fill_between(range(len(y_vals)), y_vals, alpha=0.05, color=colors_p[i])
                            
                            ax.set_facecolor('#0a0e1a')
                            fig.set_facecolor('#0a0e1a')
                            ax.set_title('Simulated Spectra from Detected Sites', color='#e2e8f0', fontsize=12)
                            ax.set_xlabel('Wavenumber Index', color='#94a3b8', fontsize=9)
                            ax.set_ylabel('Intensity', color='#94a3b8', fontsize=9)
                            ax.tick_params(colors='#94a3b8', labelsize=8)
                            ax.legend(facecolor='#0f172a', labelcolor='#e2e8f0', fontsize=8)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        with col_b:
                            # Summary table
                            summary_df = pd.DataFrame({'Polymer': predictions}).value_counts().reset_index()
                            summary_df.columns = ['Polymer', 'Count']
                            st.markdown("""
                            <div class="glass-card" style="padding: 1rem;">
                                <h4 style="margin:0 0 0.5rem 0; font-size: 0.9rem; color: #e2e8f0;">Chemical Composition</h4>
                            """, unsafe_allow_html=True)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Final Verdict
                            top_poly = summary_df['Polymer'].iloc[0]
                            st.markdown(f"""
                            <div style="background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem;">
                                <p style="margin:0; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;">Dominant Polymer</p>
                                <p style="margin:0; color: #38bdf8; font-size: 1.1rem; font-weight: 700;">{top_poly}</p>
                            </div>
                            """, unsafe_allow_html=True)

                # --- CONFIDENCE DISTRIBUTION (Always Show) ---
                if len(confidences) > 0:
                    st.markdown('<div class="section-header">📊 Detection Confidence Distribution</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.hist(confidences, bins=20, color='#0ea5e9', edgecolor='#0a0e1a', alpha=0.85)
                    ax.set_facecolor('#0a0e1a')
                    fig.set_facecolor('#0a0e1a')
                    ax.set_xlabel('Confidence Score', color='#94a3b8', fontsize=10)
                    ax.set_ylabel('Count', color='#94a3b8', fontsize=10)
                    ax.tick_params(colors='#94a3b8')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('#1e293b')
                    ax.spines['left'].set_color('#1e293b')
                    ax.axvline(x=confidences.mean(), color='#f59e0b', linestyle='--', linewidth=1.5, label=f'Mean: {confidences.mean():.2f}')
                    ax.legend(facecolor='#0a0e1a', edgecolor='#1e293b', labelcolor='#94a3b8')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        else:
            st.error("Model 'best.pt' not found. Please ensure the model file exists in the project directory.")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Drag & drop a microscope image above to start analysis
            </p>
            <p style="color: #475569; font-size: 0.8rem;">
                Supported formats: PNG, JPG, JPEG
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PAGE: SPECTRAL ANALYSIS (Raman)
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧪 Spectral Analysis":

    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2rem; font-weight: 700; margin: 0;">
            Chemical Polymer Identification
        </h1>
        <p style="color: #64748b; font-size: 0.9rem; margin: 0.3rem 0 0 0;">
            Identify polymer composition from Raman spectral signatures using a trained Random Forest classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model performance card
    st.markdown(f"""
    <div class="glass-card" style="padding: 0.8rem 1.2rem;">
        <span style="color: #94a3b8; font-size: 0.82rem;">
            <strong style="color: #38bdf8;">Algorithm:</strong> Random Forest (50 trees) &nbsp;|&nbsp;
            <strong style="color: #38bdf8;">3-Fold CV Accuracy:</strong> {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}% &nbsp;|&nbsp;
            <strong style="color: #38bdf8;">Classes:</strong> PE, PS, PMMA, PTFE, NYLON
        </span>
    </div>
    """, unsafe_allow_html=True)

    analysis_tab, insights_tab = st.tabs(["🧪 Run Prediction", "📊 Model Insights"])

    with analysis_tab:
        if st.button("🔬 Classify Random Sample", type="primary", use_container_width=False):
            sample = raman_df.sample(1)
            true_label = sample['clean_category'].values[0]
            sample_features = sample[raman_features]
            prediction = raman_model.predict(sample_features)[0]
            probabilities = raman_model.predict_proba(sample_features)[0]
            class_names = raman_model.classes_

            st.markdown("<br>", unsafe_allow_html=True)

            pred_col, chart_col = st.columns([1, 2])

            with pred_col:
                # Prediction result
                is_correct = prediction == true_label
                match_color = "#22c55e" if is_correct else "#ef4444"
                match_icon = "✅" if is_correct else "❌"
                match_text = "Correct" if is_correct else "Mismatch"

                st.markdown(f"""
                <div class="glass-card">
                    <p style="color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">
                        AI Prediction
                    </p>
                    <h2 style="color: #38bdf8; margin: 0; font-size: 1.8rem;">{prediction}</h2>
                    <div style="margin-top: 0.8rem; padding: 0.5rem; background: rgba({','.join(['34,197,94' if is_correct else '239,68,68'])}, 0.1); border-radius: 8px; text-align: center;">
                        <span style="color: {match_color}; font-size: 0.85rem; font-weight: 600;">
                            {match_icon} {match_text} — Ground Truth: {true_label}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top probabilities
                st.markdown("""
                <div class="glass-card">
                    <p style="color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.5rem 0;">
                        Class Probabilities
                    </p>
                """, unsafe_allow_html=True)
                sorted_idx = np.argsort(probabilities)[::-1]
                for idx in sorted_idx[:5]:
                    prob = probabilities[idx]
                    name = class_names[idx]
                    bar_color = "#0ea5e9" if name == prediction else "#334155"
                    st.markdown(f"""
                    <div style="margin: 0.4rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                            <span style="color: #cbd5e1; font-size: 0.8rem;">{name}</span>
                            <span style="color: #94a3b8; font-size: 0.8rem;">{prob:.1%}</span>
                        </div>
                        <div style="background: #1e293b; border-radius: 4px; height: 6px;">
                            <div style="background: {bar_color}; width: {prob*100}%; height: 6px; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with chart_col:
                # Spectral plot
                fig, ax = plt.subplots(figsize=(10, 5))
                x_vals = np.arange(min(300, len(raman_features)))
                y_vals = sample_features.values[0][:300]
                ax.fill_between(x_vals, y_vals, alpha=0.15, color='#0ea5e9')
                ax.plot(x_vals, y_vals, color='#0ea5e9', lw=1.5, alpha=0.9)
                ax.set_facecolor('#0a0e1a')
                fig.set_facecolor('#0a0e1a')
                ax.set_title(f'Raman Spectrum — Predicted: {prediction}', color='#e2e8f0', fontsize=13, fontweight='bold', pad=15)
                ax.set_xlabel('Wavenumber Index', color='#94a3b8', fontsize=10)
                ax.set_ylabel('Intensity (normalized)', color='#94a3b8', fontsize=10)
                ax.tick_params(colors='#94a3b8', labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#1e293b')
                ax.spines['left'].set_color('#1e293b')
                ax.grid(True, alpha=0.08, color='#38bdf8')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🧪</div>
                <p style="color: #94a3b8; font-size: 0.95rem;">
                    Click the button above to classify a random Raman spectrum from the database
                </p>
            </div>
            """, unsafe_allow_html=True)

    with insights_tab:
        insight_left, insight_right = st.columns(2)

        with insight_left:
            st.markdown('<div class="section-header">📊 Cross-Validation Scores</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fold_labels = [f'Fold {i+1}' for i in range(len(cv_scores))]
            bar_colors = ['#0ea5e9', '#22d3ee', '#06b6d4']
            ax.bar(fold_labels, cv_scores * 100, color=bar_colors[:len(cv_scores)], edgecolor='none', width=0.5)
            ax.axhline(y=cv_scores.mean()*100, color='#f59e0b', linestyle='--', linewidth=1.5, label=f'Mean: {cv_scores.mean()*100:.1f}%')
            ax.set_facecolor('#0a0e1a')
            fig.set_facecolor('#0a0e1a')
            ax.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=10)
            ax.tick_params(colors='#94a3b8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#1e293b')
            ax.spines['left'].set_color('#1e293b')
            ax.set_ylim(0, 105)
            ax.legend(facecolor='#0a0e1a', edgecolor='#1e293b', labelcolor='#94a3b8')
            for i, v in enumerate(cv_scores):
                ax.text(i, v*100 + 1.5, f'{v*100:.1f}%', ha='center', color='#e2e8f0', fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with insight_right:
            st.markdown('<div class="section-header">🏷️ Class Sample Counts</div>', unsafe_allow_html=True)
            class_counts = raman_df['clean_category'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_pie = ['#0ea5e9', '#22d3ee', '#38bdf8', '#06b6d4', '#67e8f9', '#a5f3fc', '#7dd3fc']
            wedges, texts, autotexts = ax.pie(
                class_counts.values, labels=class_counts.index,
                autopct='%1.1f%%', colors=colors_pie[:len(class_counts)],
                textprops={'color': '#e2e8f0', 'fontsize': 8},
                pctdistance=0.8, startangle=140
            )
            for autotext in autotexts:
                autotext.set_fontsize(7)
                autotext.set_color('#0a0e1a')
                autotext.set_fontweight('bold')
            ax.set_facecolor('#0a0e1a')
            fig.set_facecolor('#0a0e1a')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Feature importance
        st.markdown('<div class="section-header">🔑 Top 20 Feature Importances</div>', unsafe_allow_html=True)
        importances = raman_model.feature_importances_
        top_n = 20
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [raman_features[i] for i in top_indices]
        top_importances = importances[top_indices]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(top_n), top_importances, color='#0ea5e9', edgecolor='none', width=0.7)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
        ax.set_facecolor('#0a0e1a')
        fig.set_facecolor('#0a0e1a')
        ax.set_ylabel('Importance', color='#94a3b8', fontsize=10)
        ax.tick_params(colors='#94a3b8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1e293b')
        ax.spines['left'].set_color('#1e293b')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PAGE: GEOSPATIAL MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌍 Geospatial Mapping":

    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <h1 style="color: #f1f5f9; font-size: 2rem; font-weight: 700; margin: 0;">
            Global Pollution Hotspot Analysis
        </h1>
        <p style="color: #64748b; font-size: 0.9rem; margin: 0.3rem 0 0 0;">
            Aggregated geospatial data from marine research expeditions, visualized with DBSCAN density clustering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading geospatial clusters..."):
        map_data = generate_hotspot_map()

        if not map_data.empty:
            available_countries = sorted(map_data['Country'].unique().tolist())

            # Country selector in sidebar
            with st.sidebar:
                st.markdown("---")
                st.markdown("""
                <p style="color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; font-weight: 600;">
                    Region Filter
                </p>
                """, unsafe_allow_html=True)
                selected_country = st.selectbox(
                    "Filter by country",
                    options=["🌐 Global View"] + available_countries,
                    key="country_select",
                    label_visibility="collapsed"
                )

            # Filter
            if selected_country == "🌐 Global View":
                filtered_data = map_data
                display_title = "Global Overview"
            else:
                filtered_data = map_data[map_data['Country'] == selected_country]
                display_title = f"Region: {selected_country}"

            if not filtered_data.empty:

                st.markdown(f'<div class="section-header">📍 {display_title}</div>', unsafe_allow_html=True)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Sightings", f"{len(filtered_data):,}")
                cluster_count = int(filtered_data['Cluster'].max() + 1) if filtered_data['Cluster'].max() >= 0 else 0
                m2.metric("Density Clusters", cluster_count)
                m3.metric("Mean Concentration", f"{filtered_data['value'].mean():.2f}")
                m4.metric("Data Sources", filtered_data['source'].nunique())

                # Map
                st.markdown('<div class="section-header">🗺️ Spatial Distribution</div>', unsafe_allow_html=True)
                map_display = filtered_data[['Latitude', 'Longitude']].rename(
                    columns={'Latitude': 'lat', 'Longitude': 'lon'}
                )
                st.map(map_display, color='#ff4b4b', size=20)

                # Detailed breakdown
                detail_left, detail_right = st.columns(2)

                with detail_left:
                    st.markdown('<div class="section-header">📊 Source Breakdown</div>', unsafe_allow_html=True)
                    source_stats = filtered_data.groupby('source').agg({
                        'Latitude': 'count',
                        'value': ['mean', 'max', 'sum']
                    }).round(2)
                    source_stats.columns = ['Sightings', 'Avg Concentration', 'Max Concentration', 'Total Value']
                    st.dataframe(source_stats, use_container_width=True)

                with detail_right:
                    st.markdown('<div class="section-header">📈 Concentration by Source</div>', unsafe_allow_html=True)
                    source_means = filtered_data.groupby('source')['value'].mean()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors_bar = ['#0ea5e9', '#22d3ee', '#06b6d4']
                    ax.bar(source_means.index, source_means.values, color=colors_bar[:len(source_means)], edgecolor='none', width=0.5)
                    ax.set_facecolor('#0a0e1a')
                    fig.set_facecolor('#0a0e1a')
                    ax.set_ylabel('Mean Concentration', color='#94a3b8', fontsize=10)
                    ax.tick_params(colors='#94a3b8')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('#1e293b')
                    ax.spines['left'].set_color('#1e293b')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Download
                st.markdown("<br>", unsafe_allow_html=True)
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Export Filtered Data (CSV)",
                    data=csv,
                    file_name=f"microplastics_{selected_country.replace('🌐 ', '').replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"No microplastic data available for {selected_country}. Try another region.")
        else:
            st.warning("No geospatial data available. Ensure CSV files are present in the project directory.")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    EcoTrack AI — Multi-Modal Microplastic Detection & Analysis System<br>
    Machine Learning Mini Project · Pratik Patil · 2026
</div>
""", unsafe_allow_html=True)