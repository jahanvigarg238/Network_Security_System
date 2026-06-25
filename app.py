import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")

import pymongo
import pandas as pd
import streamlit as st

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.url_feature_extractor import extract_features, features_to_dataframe

# ── MongoDB client ────────────────────────────────────────────────────────────
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

database  = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base & font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp { background: #0d1117; color: #e6edf3; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* ── Cards ── */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Risk badge ── */
    .badge-safe    { background:#0d4f2e; color:#3fb950; border:1px solid #3fb950; border-radius:6px; padding:4px 12px; font-weight:600; font-size:.85rem; }
    .badge-danger  { background:#4f1010; color:#f85149; border:1px solid #f85149; border-radius:6px; padding:4px 12px; font-weight:600; font-size:.85rem; }
    .badge-warn    { background:#4f3a10; color:#e3b341; border:1px solid #e3b341; border-radius:6px; padding:4px 12px; font-weight:600; font-size:.85rem; }

    /* ── Feature row colors ── */
    .feat-safe   { border-left: 3px solid #3fb950; padding-left: 8px; margin:4px 0; font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#8b949e; }
    .feat-danger { border-left: 3px solid #f85149; padding-left: 8px; margin:4px 0; font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#8b949e; }
    .feat-warn   { border-left: 3px solid #e3b341; padding-left: 8px; margin:4px 0; font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#8b949e; }

    /* ── Risk meter bar ── */
    .risk-bar-bg { background:#21262d; border-radius:8px; height:14px; margin:8px 0; }
    .risk-bar    { height:14px; border-radius:8px; transition: width 0.4s ease; }

    /* ── Inputs ── */
    .stTextInput > div > div > input {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stFileUploader { background: #161b22; border: 1px dashed #30363d; border-radius: 12px; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: .55rem 1.4rem !important;
        font-size: .9rem !important;
        transition: opacity .2s !important;
    }
    .stButton > button:hover { opacity: .85 !important; }

    /* ── Dataframe ── */
    .stDataFrame { border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Network Security")
    st.markdown("<hr style='border-color:#30363d;margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🔗 URL Phishing Check", "📂 Batch CSV Predict", "⚙️ Train Model"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#30363d;margin:1rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#8b949e;font-size:.78rem;line-height:1.6'>
        <b style='color:#e6edf3'>How it works</b><br>
        Extracts 30+ URL features · ML model classifies phishing vs. safe · Risk score from red-flag count
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — load model (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    preprocessor = load_object("final_model/preprocessor.pkl")
    model        = load_object("final_model/model.pkl")
    return NetworkModel(preprocessor=preprocessor, model=model)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — URL PHISHING CHECK
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔗 URL Phishing Check":
    st.markdown("## 🔗 URL Phishing Detector")
    st.markdown("<p style='color:#8b949e'>Paste any URL and the model will analyse 30+ features to detect phishing.</p>", unsafe_allow_html=True)

    url = st.text_input("", placeholder="https://example.com/login?token=abc123", label_visibility="collapsed")

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        analyse = st.button("Analyse URL")

    if analyse and url.strip():
        with st.spinner("Extracting features…"):
            try:
                network_model = load_model()
                features = extract_features(url)
                df       = features_to_dataframe(features)
                y_pred   = network_model.predict(df)

                prediction  = int(y_pred[0])
                is_phishing = prediction == -1

                feature_breakdown = []
                for name, value in features.items():
                    if value == -1:
                        status = "danger"
                    elif value == 0:
                        status = "warn"
                    else:
                        status = "safe"
                    feature_breakdown.append({"name": name, "value": value, "status": status})

                red_flags  = sum(1 for f in feature_breakdown if f["status"] == "danger")
                risk_score = min(100, int((red_flags / 30) * 100) + (20 if is_phishing else 0))

                # ── Result banner ──────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                if is_phishing:
                    st.markdown(f"""
                    <div class='card' style='border-color:#f85149;'>
                        <div style='display:flex;align-items:center;gap:12px;'>
                            <span style='font-size:2rem'>🚨</span>
                            <div>
                                <div style='font-size:1.3rem;font-weight:700;color:#f85149'>Phishing Detected</div>
                                <div style='color:#8b949e;font-size:.85rem;margin-top:2px;font-family:JetBrains Mono,monospace'>{url}</div>
                            </div>
                            <span class='badge-danger' style='margin-left:auto'>MALICIOUS</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='card' style='border-color:#3fb950;'>
                        <div style='display:flex;align-items:center;gap:12px;'>
                            <span style='font-size:2rem'>✅</span>
                            <div>
                                <div style='font-size:1.3rem;font-weight:700;color:#3fb950'>URL Looks Safe</div>
                                <div style='color:#8b949e;font-size:.85rem;margin-top:2px;font-family:JetBrains Mono,monospace'>{url}</div>
                            </div>
                            <span class='badge-safe' style='margin-left:auto'>SAFE</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Risk score meter ──────────────────────────────────────
                bar_color = "#f85149" if risk_score > 60 else "#e3b341" if risk_score > 30 else "#3fb950"
                st.markdown(f"""
                <div class='card'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
                        <span style='font-weight:600;color:#e6edf3'>Risk Score</span>
                        <span style='font-weight:700;color:{bar_color};font-size:1.1rem'>{risk_score}/100</span>
                    </div>
                    <div class='risk-bar-bg'>
                        <div class='risk-bar' style='width:{risk_score}%;background:{bar_color}'></div>
                    </div>
                    <div style='color:#8b949e;font-size:.78rem;margin-top:6px'>{red_flags} red flags out of 30 features checked</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Feature breakdown ─────────────────────────────────────
                with st.expander("🔍 Feature Breakdown", expanded=False):
                    cols = st.columns(2)
                    for i, feat in enumerate(feature_breakdown):
                        label_map = {"danger": "🔴", "warn": "🟡", "safe": "🟢"}
                        with cols[i % 2]:
                            st.markdown(
                                f"<div class='feat-{feat['status']}'>{label_map[feat['status']]} {feat['name']}: <b>{feat['value']}</b></div>",
                                unsafe_allow_html=True
                            )

            except Exception as e:
                st.error(f"Error: {e}")

    elif analyse and not url.strip():
        st.warning("Please enter a URL first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH CSV PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch CSV Predict":
    st.markdown("## 📂 Batch Prediction")
    st.markdown("<p style='color:#8b949e'>Upload a CSV with URL features and get predictions for all rows at once.</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "Result" in df.columns:
                df = df.drop(columns=["Result"])

            st.markdown(f"<div style='color:#8b949e;font-size:.85rem;margin-bottom:.5rem'>📄 {uploaded.name} — {len(df)} rows, {len(df.columns)} features</div>", unsafe_allow_html=True)

            col_btn2, _ = st.columns([1, 5])
            with col_btn2:
                run_pred = st.button("Run Predictions")

            if run_pred:
                with st.spinner("Running predictions…"):
                    network_model = load_model()
                    y_pred = network_model.predict(df)
                    df["Prediction"] = y_pred
                    df["Label"] = df["Prediction"].apply(lambda x: "🚨 Phishing" if x == -1 else "✅ Safe")

                    total     = len(df)
                    phishing  = (df["Prediction"] == -1).sum()
                    safe      = total - phishing

                    # ── Summary metrics ───────────────────────────────────
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""<div class='card' style='text-align:center'>
                            <div style='font-size:1.8rem;font-weight:700;color:#e6edf3'>{total}</div>
                            <div style='color:#8b949e;font-size:.82rem'>Total URLs</div></div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class='card' style='text-align:center;border-color:#f85149'>
                            <div style='font-size:1.8rem;font-weight:700;color:#f85149'>{phishing}</div>
                            <div style='color:#8b949e;font-size:.82rem'>Phishing</div></div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class='card' style='text-align:center;border-color:#3fb950'>
                            <div style='font-size:1.8rem;font-weight:700;color:#3fb950'>{safe}</div>
                            <div style='color:#8b949e;font-size:.82rem'>Safe</div></div>""", unsafe_allow_html=True)

                    # ── Results table ─────────────────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True, height=400)

                    # ── Download ──────────────────────────────────────────
                    os.makedirs("prediction_output", exist_ok=True)
                    out_path = "prediction_output/output.csv"
                    df.to_csv(out_path, index=False)

                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="network_security_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Train Model":
    st.markdown("## ⚙️ Model Training")
    st.markdown("<p style='color:#8b949e'>Trigger the full MLOps training pipeline — data ingestion → validation → transformation → training → evaluation.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div style='font-weight:600;color:#e6edf3;margin-bottom:.5rem'>Pipeline Stages</div>
        <div style='color:#8b949e;font-size:.85rem;line-height:2'>
            1️⃣ &nbsp;Data Ingestion (MongoDB)<br>
            2️⃣ &nbsp;Data Validation (drift check)<br>
            3️⃣ &nbsp;Data Transformation<br>
            4️⃣ &nbsp;Model Training (MLflow tracking)<br>
            5️⃣ &nbsp;Model Evaluation & Export
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_t, _ = st.columns([1, 4])
    with col_t:
        start_train = st.button("🚀 Start Training")

    if start_train:
        with st.spinner("Training in progress… this may take a few minutes."):
            try:
                train_pipeline = TrainingPipeline()
                train_pipeline.run_pipeline()
                st.success("✅ Training completed successfully! Model saved to `final_model/`.")
                # Clear cached model so next prediction loads the new one
                load_model.clear()
            except Exception as e:
                st.error(f"Training failed: {e}")