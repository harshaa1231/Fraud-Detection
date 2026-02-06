import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier, StackingClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import os
import json
from datetime import datetime
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(
    page_title="AI Anomaly & Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


def hex_to_rgba(hex_color, alpha=0.1):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.8);
        --bg-glass: rgba(255, 255, 255, 0.05);
        --border-glass: rgba(255, 255, 255, 0.1);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --gradient-1: linear-gradient(135deg, #3b82f6, #8b5cf6);
        --gradient-2: linear-gradient(135deg, #10b981, #3b82f6);
        --gradient-3: linear-gradient(135deg, #f43f5e, #f59e0b);
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    p, span, label, .stMarkdown {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
    }

    .hero-section {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 24px;
        text-align: center;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8 !important;
        max-width: 600px;
        margin: 0 auto;
    }

    .metric-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }

    .metric-card.blue::before { background: var(--gradient-1); }
    .metric-card.green::before { background: var(--gradient-2); }
    .metric-card.rose::before { background: var(--gradient-3); }
    .metric-card.purple::before { background: linear-gradient(135deg, #8b5cf6, #a855f7); }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary) !important;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary) !important;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 8px;
    }

    .status-safe {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981 !important;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
    }

    .status-danger {
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.3);
        color: #f43f5e !important;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        animation: pulse-danger 2s infinite;
    }

    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.3); }
        50% { box-shadow: 0 0 20px 5px rgba(244, 63, 94, 0.15); }
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b !important;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
    }

    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #94a3b8 !important;
        font-size: 0.95rem;
    }

    .step-badge {
        display: inline-block;
        background: var(--gradient-1);
        color: white !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .nav-item {
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .nav-item:hover { background: rgba(59, 130, 246, 0.1); }
    .nav-item.active { background: rgba(59, 130, 246, 0.2); border-left: 3px solid #3b82f6; }

    .stButton > button {
        background: var(--gradient-1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        opacity: 0.9 !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }

    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        padding: 8px 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
        color: var(--text-primary) !important;
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }

    .stSlider > div > div {
        color: var(--text-secondary) !important;
    }

    .stCheckbox > label {
        color: var(--text-secondary) !important;
    }

    .stExpander {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
    }

    .stProgress > div > div > div {
        background: var(--gradient-1) !important;
    }

    div[data-testid="stFileUploader"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
    }

    .risk-meter {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }

    .risk-score {
        font-size: 3rem;
        font-weight: 800;
    }

    .risk-low { color: #10b981 !important; }
    .risk-medium { color: #f59e0b !important; }
    .risk-high { color: #f43f5e !important; }

    .feature-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #93c5fd !important;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
        margin: 24px 0;
    }

    .stDownloadButton > button {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #10b981 !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(16, 185, 129, 0.3) !important;
    }

    .dash-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .score-ring {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 12px;
        position: relative;
    }

    .score-ring-inner {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: var(--bg-primary);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }

    .score-ring-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary) !important;
    }

    .score-ring-label {
        font-size: 0.7rem;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .model-rank {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 18px;
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }

    .model-rank:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateX(4px);
    }

    .rank-badge {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.9rem;
        flex-shrink: 0;
    }

    .rank-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #000 !important; }
    .rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: #000 !important; }
    .rank-3 { background: linear-gradient(135deg, #b45309, #92400e); color: #fff !important; }
    .rank-other { background: rgba(255,255,255,0.1); color: var(--text-secondary) !important; }

    .model-rank-name {
        flex: 1;
        font-weight: 600;
        color: var(--text-primary) !important;
        font-size: 0.95rem;
    }

    .model-rank-score {
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--accent-blue) !important;
    }

    .qc-scenario-btn {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        color: var(--text-primary) !important;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }

    .qc-scenario-btn:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
        background: rgba(59, 130, 246, 0.1) !important;
    }

    .risk-gauge-container {
        position: relative;
        width: 220px;
        height: 130px;
        margin: 0 auto 16px;
        overflow: hidden;
    }

    .verdict-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .verdict-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
    }

    .verdict-safe::after { background: var(--gradient-2); }
    .verdict-warning::after { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .verdict-danger::after { background: var(--gradient-3); }

    .model-verdict-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 10px;
        margin-bottom: 6px;
    }

    .verdict-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .verdict-dot-safe { background: #10b981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
    .verdict-dot-danger { background: #f43f5e; box-shadow: 0 0 8px rgba(244,63,94,0.5); }

    .insight-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
    }

    .insight-good { background: rgba(16,185,129,0.15); color: #10b981 !important; border: 1px solid rgba(16,185,129,0.3); }
    .insight-warn { background: rgba(245,158,11,0.15); color: #f59e0b !important; border: 1px solid rgba(245,158,11,0.3); }
    .insight-bad { background: rgba(244,63,94,0.15); color: #f43f5e !important; border: 1px solid rgba(244,63,94,0.3); }
    </style>
    """, unsafe_allow_html=True)


def friendly_feature_name(raw_name):
    name = str(raw_name)
    import re
    if re.match(r'^V\d+$', name) or re.match(r'^PC\d+$', name, re.IGNORECASE):
        return f"Signal {name}"
    if re.match(r'^x\d+$', name, re.IGNORECASE):
        return f"Feature {name}"
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return name.strip().title()


def is_dummy_feature(feature_name):
    name_lower = feature_name.lower()
    if '_' in feature_name:
        parts = feature_name.rsplit('_', 1)
        if len(parts) == 2 and (parts[1].replace(' ', '').replace('-', '').isalnum()):
            if any(kw in name_lower for kw in ['_yes', '_no', '_true', '_false', '_male', '_female']):
                return True
    return False


def get_feature_widget_config(feature_name, mean_val, std_val):
    name_lower = feature_name.lower()
    friendly = friendly_feature_name(feature_name)
    std_val = max(std_val, 1e-6)

    if is_dummy_feature(feature_name):
        return {
            'type': 'toggle', 'icon': '🏷️', 'label': friendly,
            'help': f'Is this {friendly}? (Yes or No)',
            'mean': mean_val, 'std': std_val
        }

    if any(kw in name_lower for kw in ['amount', 'price', 'cost', 'value', 'salary', 'income', 'revenue', 'payment', 'balance', 'total', 'fee', 'charge']):
        low = max(0.0, mean_val - 2 * std_val)
        high = mean_val + 4 * std_val
        return {
            'type': 'dollar', 'icon': '💰', 'label': friendly,
            'help': 'How much money was involved?',
            'min': round(low, 2), 'max': round(high, 2),
            'default': round(mean_val, 2), 'step': round(max(0.01, std_val / 20), 2)
        }

    if any(kw in name_lower for kw in ['_hour', 'hour_of', 'time_hour']):
        return {
            'type': 'hour', 'icon': '🕐', 'label': friendly,
            'help': 'What time of day?',
            'options': {f'{h}:00 {"AM" if h < 12 else "PM"}' if h != 0 else '12:00 AM': h for h in range(24)}
        }

    if any(kw in name_lower for kw in ['dayofweek', 'day_of_week', 'weekday']):
        return {
            'type': 'day', 'icon': '📅', 'label': friendly,
            'help': 'What day of the week?',
            'options': {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        }

    if 'month' in name_lower and any(kw in name_lower for kw in ['_month', 'month_']):
        return {
            'type': 'month', 'icon': '📅', 'label': friendly,
            'help': 'What month?',
            'options': {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        }

    if any(kw in name_lower for kw in ['transaction_time', 'elapsed_time', 'time_since', 'seconds_since']):
        hours = mean_val / 3600
        return {
            'type': 'time_elapsed', 'icon': '⏱️', 'label': friendly,
            'help': f'Time since the start of the observation (typically measured in hours)',
            'min': max(0.0, mean_val - 2 * std_val), 'max': mean_val + 3 * std_val,
            'default': mean_val, 'step': round(std_val / 10, 2),
            'hours_equiv': hours
        }

    if any(kw in name_lower for kw in ['frequency', 'count', 'times', 'number_of', 'quantity', 'n_', 'num_', 'repeat']):
        return {
            'type': 'count', 'icon': '🔢', 'label': friendly,
            'help': 'How many times did this happen?',
            'min': max(0, round(mean_val - 2 * std_val)), 'max': round(mean_val + 4 * std_val),
            'default': round(mean_val, 1), 'step': round(max(0.1, std_val / 10), 1)
        }

    if any(kw in name_lower for kw in ['risk', 'danger', 'threat', 'suspicious']):
        return {
            'type': 'risk_slider', 'icon': '⚠️', 'label': friendly,
            'help': 'How risky does this seem?',
            'low_text': 'Low Risk', 'high_text': 'High Risk',
            'mean': mean_val, 'std': std_val
        }

    if any(kw in name_lower for kw in ['trust', 'confidence', 'reliability', 'verified', 'legitimate']):
        return {
            'type': 'trust_slider', 'icon': '🛡️', 'label': friendly,
            'help': 'How trustworthy or verified is this?',
            'low_text': 'Not Trusted', 'high_text': 'Fully Trusted',
            'mean': mean_val, 'std': std_val
        }

    if any(kw in name_lower for kw in ['pattern', 'behavior', 'spending', 'activity', 'usage']):
        return {
            'type': 'pattern_slider', 'icon': '📊', 'label': friendly,
            'help': 'How does this compare to normal behavior?',
            'low_text': 'Below Normal', 'high_text': 'Above Normal',
            'mean': mean_val, 'std': std_val
        }

    if any(kw in name_lower for kw in ['score', 'rating', 'index', 'level']):
        return {
            'type': 'score_slider', 'icon': '📈', 'label': friendly,
            'help': 'Score level',
            'low_text': 'Low', 'high_text': 'High',
            'mean': mean_val, 'std': std_val
        }

    if 'age' in name_lower:
        return {
            'type': 'number', 'icon': '👤', 'label': friendly,
            'help': 'Enter the age',
            'min': max(0, mean_val - 3 * std_val), 'max': mean_val + 3 * std_val,
            'default': mean_val, 'step': 1.0
        }

    if any(kw in name_lower for kw in ['distance', 'miles', 'km', 'meters']):
        return {
            'type': 'number', 'icon': '📍', 'label': friendly,
            'help': 'How far? (distance)',
            'min': max(0, mean_val - 2 * std_val), 'max': mean_val + 4 * std_val,
            'default': mean_val, 'step': round(max(0.1, std_val / 10), 1)
        }

    return {
        'type': 'number', 'icon': '📋', 'label': friendly,
        'help': f'Value for {friendly}',
        'min': mean_val - 3 * std_val, 'max': mean_val + 3 * std_val,
        'default': mean_val, 'step': round(max(0.01, std_val / 10), 2)
    }


def render_smart_input(feature_name, config, current_val, key_prefix="qc"):
    key = f"{key_prefix}_{feature_name}"
    wtype = config['type']

    if wtype == 'toggle':
        val = st.selectbox(
            f"{config['icon']} {config['label']}",
            options=['No', 'Yes'], index=1 if current_val > 0.5 else 0,
            help=config['help'], key=key
        )
        return 1.0 if val == 'Yes' else 0.0

    elif wtype == 'dollar':
        val = st.number_input(
            f"{config['icon']} {config['label']} ($)",
            min_value=config['min'], max_value=config['max'],
            value=float(np.clip(current_val, config['min'], config['max'])),
            step=config['step'], help=config['help'], key=key
        )
        return val

    elif wtype == 'hour':
        options = list(config['options'].keys())
        current_hour = int(np.clip(round(current_val), 0, 23))
        val = st.selectbox(
            f"{config['icon']} {config['label']}",
            options=options, index=current_hour,
            help=config['help'], key=key
        )
        return float(config['options'][val])

    elif wtype == 'day':
        options = list(config['options'].keys())
        current_day = int(np.clip(round(current_val), 0, 6))
        val = st.selectbox(
            f"{config['icon']} {config['label']}",
            options=options, index=current_day,
            help=config['help'], key=key
        )
        return float(config['options'][val])

    elif wtype == 'month':
        options = list(config['options'].keys())
        current_month = int(np.clip(round(current_val), 1, 12)) - 1
        val = st.selectbox(
            f"{config['icon']} {config['label']}",
            options=options, index=current_month,
            help=config['help'], key=key
        )
        return float(config['options'][val])

    elif wtype == 'time_elapsed':
        val = st.number_input(
            f"{config['icon']} {config['label']}",
            min_value=config['min'], max_value=config['max'],
            value=float(np.clip(current_val, config['min'], config['max'])),
            step=config['step'], help=config['help'], key=key
        )
        return val

    elif wtype == 'count':
        val = st.number_input(
            f"{config['icon']} {config['label']}",
            min_value=float(config['min']), max_value=float(config['max']),
            value=float(np.clip(current_val, config['min'], config['max'])),
            step=float(config['step']), help=config['help'], key=key
        )
        return val

    elif wtype in ('risk_slider', 'trust_slider', 'pattern_slider', 'score_slider'):
        mean = config['mean']
        std = config['std']
        low = mean - 3 * std
        high = mean + 3 * std
        st.markdown(f"{config['icon']} **{config['label']}**")
        col_l, col_s, col_r = st.columns([1, 6, 1])
        with col_l:
            st.markdown(f"<div style='font-size:0.7rem;color:#64748b;text-align:center;margin-top:8px;'>{config['low_text']}</div>", unsafe_allow_html=True)
        with col_s:
            val = st.slider(
                config['label'], min_value=float(low), max_value=float(high),
                value=float(np.clip(current_val, low, high)),
                step=float(max(0.01, std / 20)), label_visibility="collapsed",
                help=config['help'], key=key
            )
        with col_r:
            st.markdown(f"<div style='font-size:0.7rem;color:#64748b;text-align:center;margin-top:8px;'>{config['high_text']}</div>", unsafe_allow_html=True)
        return val

    else:
        min_v = config.get('min', current_val - 10)
        max_v = config.get('max', current_val + 10)
        step = config.get('step', 0.1)
        val = st.number_input(
            f"{config['icon']} {config['label']}",
            min_value=float(min_v), max_value=float(max_v),
            value=float(np.clip(current_val, min_v, max_v)),
            step=float(step), help=config['help'], key=key
        )
        return val


def generate_sample_data(n_samples=10000, fraud_ratio=0.02):
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    normal_amount = np.random.gamma(2, 50, n_normal)
    normal_time = np.random.uniform(0, 172800, n_normal)
    normal_v1 = np.random.normal(0, 1, n_normal)
    normal_v2 = np.random.normal(0, 1, n_normal)
    normal_v3 = np.random.normal(0, 1, n_normal)
    normal_v4 = np.random.normal(0, 1, n_normal)

    fraud_amount = np.random.gamma(5, 100, n_fraud)
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    fraud_v1 = np.random.normal(2, 1.5, n_fraud)
    fraud_v2 = np.random.normal(-2, 1.5, n_fraud)
    fraud_v3 = np.random.normal(3, 2, n_fraud)
    fraud_v4 = np.random.normal(-1.5, 1.8, n_fraud)

    data = pd.DataFrame({
        'Transaction_Time': np.concatenate([normal_time, fraud_time]),
        'Transaction_Amount': np.concatenate([normal_amount, fraud_amount]),
        'Spending_Pattern': np.concatenate([normal_v1, fraud_v1]),
        'Location_Risk': np.concatenate([normal_v2, fraud_v2]),
        'Frequency_Score': np.concatenate([normal_v3, fraud_v3]),
        'Device_Trust': np.concatenate([normal_v4, fraud_v4]),
        'Is_Fraud': np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    })

    data = data.sample(frac=1).reset_index(drop=True)
    return data


def infer_schema(df):
    schema = {
        'numeric_cols': [],
        'categorical_cols': [],
        'datetime_cols': [],
        'id_cols': [],
        'binary_cols': [],
        'target_candidates': [],
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'missing_pct': {}
    }

    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        schema['missing_pct'][col] = round(missing_pct, 1)

        nunique = df[col].nunique()

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            schema['datetime_cols'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if nunique == 2:
                schema['binary_cols'].append(col)
                schema['target_candidates'].append(col)
            elif nunique <= 1 or (nunique / len(df) > 0.95 and nunique > 100):
                schema['id_cols'].append(col)
            else:
                schema['numeric_cols'].append(col)
        else:
            if nunique == 2:
                schema['binary_cols'].append(col)
                schema['target_candidates'].append(col)
            elif nunique <= 20:
                schema['categorical_cols'].append(col)
            elif nunique / len(df) > 0.5:
                schema['id_cols'].append(col)
            else:
                schema['categorical_cols'].append(col)

    return schema


def smart_preprocess(df, target_column=None, schema=None):
    if schema is None:
        schema = infer_schema(df)

    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column].copy()

        if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    else:
        X = df.copy()
        y = None

    drop_cols = [c for c in schema['id_cols'] if c in X.columns]
    X = X.drop(columns=drop_cols, errors='ignore')

    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            try:
                X[col + '_hour'] = X[col].dt.hour
                X[col + '_dayofweek'] = X[col].dt.dayofweek
                X[col + '_month'] = X[col].dt.month
            except Exception:
                pass
            X = X.drop(columns=[col])
        elif X[col].dtype == 'object':
            try:
                parsed = pd.to_datetime(X[col], errors='coerce', infer_datetime_format=True)
                if parsed.notna().mean() > 0.5:
                    X[col + '_hour'] = parsed.dt.hour.fillna(0).astype(int)
                    X[col + '_dayofweek'] = parsed.dt.dayofweek.fillna(0).astype(int)
                    X[col + '_month'] = parsed.dt.month.fillna(0).astype(int)
                    X = X.drop(columns=[col])
                    continue
            except Exception:
                pass

    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for col in cat_cols:
        X[col] = X[col].fillna('Unknown')
        if X[col].nunique() <= 15:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    for col in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(0)
            except Exception:
                X = X.drop(columns=[col])

    if y is not None:
        can_stratify = y.nunique() >= 2 and y.value_counts().min() >= 2
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42,
                stratify=y if can_stratify else None
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
    else:
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        y_train, y_test = None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler


def train_supervised_models(X_train, X_test, y_train, y_test, use_smote=False, progress_callback=None):
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

    results = {}
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(f"Training {name}...", (idx + 1) / len(models))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    return results


def train_anomaly_models(X_train, X_test, y_train, y_test, contamination=0.02, progress_callback=None):
    if y_train is not None:
        X_train_normal = X_train[y_train == 0]
    else:
        X_train_normal = X_train

    models = {
        'Isolation Forest': IsolationForest(contamination=contamination, random_state=42, n_jobs=-1),
        'One-Class SVM': OneClassSVM(nu=contamination, kernel='rbf', gamma='auto'),
    }

    results = {}
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(f"Training {name}...", (idx + 1) / (len(models) + 1))
        model.fit(X_train_normal)
        y_pred = model.predict(X_test)
        y_pred_binary = np.where(y_pred == -1, 1, 0)

        if hasattr(model, 'decision_function'):
            anomaly_scores = -model.decision_function(X_test)
        else:
            anomaly_scores = -model.score_samples(X_test)

        metrics = {
            'model': model,
            'predictions': y_pred_binary,
            'anomaly_scores': anomaly_scores,
        }

        if y_test is not None:
            metrics.update({
                'accuracy': accuracy_score(y_test, y_pred_binary),
                'precision': precision_score(y_test, y_pred_binary, zero_division=0),
                'recall': recall_score(y_test, y_pred_binary, zero_division=0),
                'f1': f1_score(y_test, y_pred_binary, zero_division=0),
                'roc_auc': roc_auc_score(y_test, anomaly_scores),
                'confusion_matrix': confusion_matrix(y_test, y_pred_binary)
            })

        results[name] = metrics

    if progress_callback:
        progress_callback("Training Local Outlier Factor...", 1.0)
    lof = LocalOutlierFactor(contamination=contamination, novelty=True, n_jobs=-1)
    lof.fit(X_train_normal)
    y_pred_lof = lof.predict(X_test)
    y_pred_lof_binary = np.where(y_pred_lof == -1, 1, 0)
    anomaly_scores_lof = -lof.decision_function(X_test)

    lof_metrics = {
        'model': lof,
        'predictions': y_pred_lof_binary,
        'anomaly_scores': anomaly_scores_lof,
    }

    if y_test is not None:
        lof_metrics.update({
            'accuracy': accuracy_score(y_test, y_pred_lof_binary),
            'precision': precision_score(y_test, y_pred_lof_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_lof_binary, zero_division=0),
            'f1': f1_score(y_test, y_pred_lof_binary, zero_division=0),
            'roc_auc': roc_auc_score(y_test, anomaly_scores_lof),
            'confusion_matrix': confusion_matrix(y_test, y_pred_lof_binary)
        })

    results['Local Outlier Factor'] = lof_metrics
    return results


def train_ensemble_model(X_train, X_test, y_train, y_test, use_smote=False, progress_callback=None):
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    base_estimators = [
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'))
    ]

    if progress_callback:
        progress_callback("Training Voting Ensemble...", 0.25)
    voting_hard = VotingClassifier(estimators=base_estimators, voting='hard', n_jobs=-1)
    voting_hard.fit(X_train, y_train)
    y_pred_hard = voting_hard.predict(X_test)

    if progress_callback:
        progress_callback("Training Soft Voting...", 0.5)
    voting_soft = VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=-1)
    voting_soft.fit(X_train, y_train)
    y_pred_soft = voting_soft.predict(X_test)
    y_proba_soft = voting_soft.predict_proba(X_test)[:, 1]

    if progress_callback:
        progress_callback("Training Stacking Ensemble...", 0.75)
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=3, n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_test)
    y_proba_stack = stacking.predict_proba(X_test)[:, 1]

    if progress_callback:
        progress_callback("Training Hybrid Model...", 1.0)
    iso_forest = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    iso_scores_train = iso_forest.decision_function(X_train).reshape(-1, 1)
    iso_scores_test = iso_forest.decision_function(X_test).reshape(-1, 1)
    X_train_hybrid = np.hstack([X_train, iso_scores_train])
    X_test_hybrid = np.hstack([X_test, iso_scores_test])
    hybrid_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    hybrid_model.fit(X_train_hybrid, y_train)
    y_pred_hybrid = hybrid_model.predict(X_test_hybrid)
    y_proba_hybrid = hybrid_model.predict_proba(X_test_hybrid)[:, 1]

    results = {
        'Voting (Hard)': {
            'model': voting_hard,
            'accuracy': accuracy_score(y_test, y_pred_hard),
            'precision': precision_score(y_test, y_pred_hard, zero_division=0),
            'recall': recall_score(y_test, y_pred_hard, zero_division=0),
            'f1': f1_score(y_test, y_pred_hard, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_hard),
            'predictions': y_pred_hard,
            'probabilities': y_pred_hard.astype(float),
            'confusion_matrix': confusion_matrix(y_test, y_pred_hard)
        },
        'Voting (Soft)': {
            'model': voting_soft,
            'accuracy': accuracy_score(y_test, y_pred_soft),
            'precision': precision_score(y_test, y_pred_soft, zero_division=0),
            'recall': recall_score(y_test, y_pred_soft, zero_division=0),
            'f1': f1_score(y_test, y_pred_soft, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_soft),
            'predictions': y_pred_soft,
            'probabilities': y_proba_soft,
            'confusion_matrix': confusion_matrix(y_test, y_pred_soft)
        },
        'Stacking': {
            'model': stacking,
            'accuracy': accuracy_score(y_test, y_pred_stack),
            'precision': precision_score(y_test, y_pred_stack, zero_division=0),
            'recall': recall_score(y_test, y_pred_stack, zero_division=0),
            'f1': f1_score(y_test, y_pred_stack, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_stack),
            'predictions': y_pred_stack,
            'probabilities': y_proba_stack,
            'confusion_matrix': confusion_matrix(y_test, y_pred_stack)
        },
        'Hybrid (Supervised + Anomaly)': {
            'model': hybrid_model,
            'iso_forest': iso_forest,
            'accuracy': accuracy_score(y_test, y_pred_hybrid),
            'precision': precision_score(y_test, y_pred_hybrid, zero_division=0),
            'recall': recall_score(y_test, y_pred_hybrid, zero_division=0),
            'f1': f1_score(y_test, y_pred_hybrid, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_hybrid),
            'predictions': y_pred_hybrid,
            'probabilities': y_proba_hybrid,
            'confusion_matrix': confusion_matrix(y_test, y_pred_hybrid)
        }
    }
    return results


def train_autoencoder(X_train, X_test, y_train, y_test, encoding_dim=None, epochs=50, threshold_percentile=95, progress_callback=None):
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel('ERROR')

    if y_train is not None:
        X_train_normal = X_train[y_train == 0]
    else:
        X_train_normal = X_train

    n_features = X_train.shape[1]
    if encoding_dim is None:
        encoding_dim = max(2, n_features // 3)

    encoder_input = keras.layers.Input(shape=(n_features,))
    x = keras.layers.Dense(n_features * 2, activation='relu')(encoder_input)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(encoding_dim * 2, activation='relu')(x)
    encoded = keras.layers.Dense(encoding_dim, activation='relu')(x)
    x = keras.layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    x = keras.layers.Dense(n_features * 2, activation='relu')(x)
    decoded = keras.layers.Dense(n_features, activation='linear')(x)

    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    progress_bar = st.progress(0)
    status_text = st.empty()

    class StreamlitCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress(min((epoch + 1) / epochs, 1.0))
            status_text.text(f"Training deep learning model... Epoch {epoch + 1}/{epochs}")

    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=epochs, batch_size=32, validation_split=0.15, verbose=0,
        callbacks=[StreamlitCallback(), keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    train_reconstructions = autoencoder.predict(X_train_normal, verbose=0)
    train_mse = np.mean(np.power(X_train_normal - train_reconstructions, 2), axis=1)
    threshold = np.percentile(train_mse, threshold_percentile)

    test_reconstructions = autoencoder.predict(X_test, verbose=0)
    test_mse = np.mean(np.power(X_test - test_reconstructions, 2), axis=1)
    y_pred = (test_mse > threshold).astype(int)

    metrics = {
        'Autoencoder': {
            'model': autoencoder,
            'predictions': y_pred,
            'anomaly_scores': test_mse,
            'threshold': threshold,
            'history': history.history,
            'train_mse': train_mse
        }
    }

    if y_test is not None:
        metrics['Autoencoder'].update({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, test_mse),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        })

    status_text.text("Deep learning model trained!")
    return metrics


def get_risk_level(score):
    if score >= 0.7:
        return "HIGH RISK", "risk-high", "status-danger"
    elif score >= 0.4:
        return "MEDIUM RISK", "risk-medium", "status-warning"
    else:
        return "LOW RISK", "risk-low", "status-safe"


def explain_metric(name, value):
    explanations = {
        'accuracy': f"The model correctly identified {value:.1%} of all transactions",
        'precision': f"When the model flags something as suspicious, it's right {value:.1%} of the time",
        'recall': f"The model catches {value:.1%} of all actual suspicious cases",
        'f1': f"Overall balance between catching fraud and avoiding false alarms: {value:.1%}",
        'roc_auc': f"The model's ability to tell apart good and bad transactions: {value:.1%}"
    }
    return explanations.get(name, f"{name}: {value:.4f}")


def render_metric_card(icon, value, label, color_class="blue"):
    st.markdown(f"""
    <div class="metric-card {color_class}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def plotly_dark_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color='#f1f5f9', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
        hovermode='x unified',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def save_models(model_name, results, scaler, feature_names, preprocessed_data=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    for name, res in results.items():
        model = res['model']
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        if hasattr(model, 'save'):
            model.save(os.path.join(save_dir, f"{safe_name}_model.keras"))
        else:
            joblib.dump(model, os.path.join(save_dir, f"{safe_name}_model.joblib"))
        if 'iso_forest' in res:
            joblib.dump(res['iso_forest'], os.path.join(save_dir, f"{safe_name}_iso_forest.joblib"))

    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(list(feature_names), os.path.join(save_dir, "feature_names.joblib"))

    metadata = {
        'timestamp': timestamp,
        'model_type': model_name,
        'models': list(results.keys()),
        'n_features': len(feature_names),
        'feature_names': list(feature_names),
        'metrics': {},
        'artifacts': {}
    }
    for name, res in results.items():
        if 'accuracy' in res:
            metadata['metrics'][name] = {
                'accuracy': float(res['accuracy']),
                'precision': float(res['precision']),
                'recall': float(res['recall']),
                'f1': float(res['f1']),
                'roc_auc': float(res['roc_auc'])
            }
        if 'iso_forest' in res:
            metadata['artifacts'][name] = {'has_iso_forest': True}
        if 'threshold' in res:
            if name not in metadata['artifacts']:
                metadata['artifacts'][name] = {}
            metadata['artifacts'][name]['threshold'] = float(res['threshold'])

    with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    return save_dir


def load_saved_models():
    saved = []
    if not os.path.exists(MODELS_DIR):
        return saved
    for dirname in sorted(os.listdir(MODELS_DIR), reverse=True):
        meta_path = os.path.join(MODELS_DIR, dirname, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            metadata['dir_name'] = dirname
            metadata['dir_path'] = os.path.join(MODELS_DIR, dirname)
            saved.append(metadata)
    return saved


def load_model_from_dir(dir_path, metadata):
    scaler = joblib.load(os.path.join(dir_path, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(dir_path, "feature_names.joblib"))
    models = {}
    extras = {}
    artifacts = metadata.get('artifacts', {})

    for name in metadata['models']:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        keras_path = os.path.join(dir_path, f"{safe_name}_model.keras")
        joblib_path = os.path.join(dir_path, f"{safe_name}_model.joblib")
        if os.path.exists(keras_path):
            from tensorflow import keras
            models[name] = keras.models.load_model(keras_path)
        elif os.path.exists(joblib_path):
            models[name] = joblib.load(joblib_path)

        model_extras = {}
        iso_path = os.path.join(dir_path, f"{safe_name}_iso_forest.joblib")
        if os.path.exists(iso_path):
            model_extras['iso_forest'] = joblib.load(iso_path)
        if name in artifacts and 'threshold' in artifacts[name]:
            model_extras['threshold'] = artifacts[name]['threshold']
        if model_extras:
            extras[name] = model_extras

    return models, scaler, feature_names, extras


def generate_report_html(all_results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html><head><title>Analysis Report - {timestamp}</title>
<style>
body {{ font-family: 'Inter', 'Segoe UI', Arial, sans-serif; margin: 0; background: #0a0e1a; color: #f1f5f9; }}
.container {{ max-width: 900px; margin: 0 auto; padding: 40px; }}
.header {{ background: linear-gradient(135deg, rgba(59,130,246,0.3), rgba(139,92,246,0.3)); padding: 40px; border-radius: 16px; margin-bottom: 30px; text-align: center; }}
.header h1 {{ margin: 0; font-size: 28px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.header p {{ margin: 8px 0 0; color: #94a3b8; }}
.card {{ background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 24px; margin-bottom: 20px; }}
.card h2 {{ color: #f1f5f9; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th {{ background: rgba(59,130,246,0.2); color: #93c5fd; padding: 12px 15px; text-align: left; border-radius: 8px; }}
td {{ padding: 10px 15px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #f1f5f9; }}
.good {{ color: #10b981; font-weight: 600; }}
.ok {{ color: #f59e0b; font-weight: 600; }}
.bad {{ color: #f43f5e; font-weight: 600; }}
.summary-box {{ display: inline-block; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2); padding: 16px 28px; border-radius: 12px; margin: 6px; text-align: center; }}
.summary-box .value {{ font-size: 24px; font-weight: 800; color: #f1f5f9; }}
.summary-box .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
</style></head><body><div class="container">
<div class="header"><h1>AI Anomaly & Fraud Detection Report</h1><p>Generated: {timestamp}</p></div>"""

    if all_results:
        results_with_metrics = {k: v for k, v in all_results.items() if 'f1' in v}
        if results_with_metrics:
            best_f1 = max(results_with_metrics.items(), key=lambda x: x[1]['f1'])
            best_auc = max(results_with_metrics.items(), key=lambda x: x[1]['roc_auc'])
            html += f"""<div class="card"><h2>Summary</h2>
            <div class="summary-box"><div class="value">{len(all_results)}</div><div class="label">Models Trained</div></div>
            <div class="summary-box"><div class="value">{best_f1[0]}</div><div class="label">Best Model</div></div>
            <div class="summary-box"><div class="value">{best_f1[1]['f1']:.4f}</div><div class="label">Best Score</div></div>
            <div class="summary-box"><div class="value">{best_auc[1]['roc_auc']:.4f}</div><div class="label">Best Detection Rate</div></div></div>"""

            html += '<div class="card"><h2>All Models</h2><table>'
            html += '<tr><th>Model</th><th>Overall Accuracy</th><th>Alert Quality</th><th>Catch Rate</th><th>Balance Score</th><th>Detection Power</th></tr>'
            for name, res in results_with_metrics.items():
                def cc(v):
                    return 'good' if v >= 0.9 else ('ok' if v >= 0.7 else 'bad')
                html += f'<tr><td><strong>{name}</strong></td>'
                for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    v = res[m]
                    html += f'<td class="{cc(v)}">{v:.4f}</td>'
                html += '</tr>'
            html += '</table></div>'

    html += '</div></body></html>'
    return html


def page_home():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">AI Anomaly & Fraud Detector</div>
        <p class="hero-subtitle">Upload any dataset and let our AI find hidden patterns, anomalies, and suspicious activity — no technical skills needed</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("📊", "Any Data", "Upload Any Dataset", "blue")
    with col2:
        render_metric_card("🤖", "10+ Models", "AI-Powered Analysis", "purple")
    with col3:
        render_metric_card("🔍", "Auto-Detect", "Finds Hidden Patterns", "green")
    with col4:
        render_metric_card("✅", "Easy Check", "Verify Any Transaction", "rose")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Choose Your Path")
    st.markdown("Pick the option that fits you best. Both paths use the same powerful AI — they just differ in how much control you get.")

    col_quick, col_full = st.columns(2)

    with col_quick:
        st.markdown("""
        <div class="glass-card" style="border: 1px solid rgba(99, 102, 241, 0.3); min-height: 420px;">
            <div style="text-align: center; margin-bottom: 16px;">
                <span style="font-size: 2.5rem;">🚀</span>
            </div>
            <h3 style="color: #818cf8 !important; text-align: center; margin-bottom: 8px;">Quick Check</h3>
            <div style="text-align: center; margin-bottom: 16px;">
                <span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">BEST FOR NON-TECHNICAL USERS</span>
            </div>
            <p style="color: #94a3b8 !important; font-size: 0.92rem; line-height: 1.7;">
                Just want to see what the AI can do? This path loads a sample dataset, trains the AI for you automatically, and takes you straight to a simple form where you can test different scenarios.
            </p>
            <div style="margin-top: 14px; color: #c8d6e5 !important; font-size: 0.85rem; line-height: 2;">
                <strong>What you'll do:</strong><br>
                ✓ Click one button — AI sets everything up<br>
                ✓ Fill in a simple form (dollar amounts, times, etc.)<br>
                ✓ See a clear risk score: safe, suspicious, or dangerous<br>
                ✓ No data files or technical knowledge needed
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Quick Check  →", type="primary", key="home_quick_check", use_container_width=True):
            st.session_state['nav_target'] = '🔍 Quick Check'
            st.rerun()

    with col_full:
        st.markdown("""
        <div class="glass-card" style="border: 1px solid rgba(16, 185, 129, 0.3); min-height: 420px;">
            <div style="text-align: center; margin-bottom: 16px;">
                <span style="font-size: 2.5rem;">🔬</span>
            </div>
            <h3 style="color: #34d399 !important; text-align: center; margin-bottom: 8px;">Full Analysis</h3>
            <div style="text-align: center; margin-bottom: 16px;">
                <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">BEST FOR DATA / TECHNICAL USERS</span>
            </div>
            <p style="color: #94a3b8 !important; font-size: 0.92rem; line-height: 1.7;">
                Bring your own CSV or Excel file and take full control. Choose which AI models to train, tune settings, explore detailed charts and metrics, and download professional reports.
            </p>
            <div style="margin-top: 14px; color: #c8d6e5 !important; font-size: 0.85rem; line-height: 2;">
                <strong>What you'll do:</strong><br>
                ✓ Upload your own dataset (CSV / Excel)<br>
                ✓ Pick and configure 10+ AI models<br>
                ✓ Explore interactive charts, ROC curves, radar plots<br>
                ✓ Download detailed reports and save trained models
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Upload My Data  →", type="secondary", key="home_full_analysis", use_container_width=True):
            st.session_state['nav_target'] = '📂 Upload Data'
            st.rerun()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### How the Two Paths Compare")
    st.markdown("""
    <div class="glass-card" style="padding: 20px 28px;">
        <table style="width: 100%; border-collapse: collapse; color: #c8d6e5;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.2);">
                    <th style="text-align: left; padding: 10px 8px; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;"></th>
                    <th style="text-align: center; padding: 10px 8px; color: #818cf8; font-size: 0.85rem;">🚀 Quick Check</th>
                    <th style="text-align: center; padding: 10px 8px; color: #34d399; font-size: 0.85rem;">🔬 Full Analysis</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <td style="padding: 10px 8px; color: #94a3b8;">Data</td>
                    <td style="text-align: center; padding: 10px 8px;">Sample dataset provided</td>
                    <td style="text-align: center; padding: 10px 8px;">Upload your own files</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <td style="padding: 10px 8px; color: #94a3b8;">Setup</td>
                    <td style="text-align: center; padding: 10px 8px;">One click — fully automatic</td>
                    <td style="text-align: center; padding: 10px 8px;">You choose models & settings</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <td style="padding: 10px 8px; color: #94a3b8;">Results</td>
                    <td style="text-align: center; padding: 10px 8px;">Simple risk score & verdict</td>
                    <td style="text-align: center; padding: 10px 8px;">Charts, metrics & reports</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <td style="padding: 10px 8px; color: #94a3b8;">Technical skill</td>
                    <td style="text-align: center; padding: 10px 8px;">None needed</td>
                    <td style="text-align: center; padding: 10px 8px;">Some data knowledge helps</td>
                </tr>
                <tr>
                    <td style="padding: 10px 8px; color: #94a3b8;">Time</td>
                    <td style="text-align: center; padding: 10px 8px;">~1 minute</td>
                    <td style="text-align: center; padding: 10px 8px;">5–15 minutes</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    if 'df' in st.session_state:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Current Dataset")
        df = st.session_state['df']
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("📋", f"{len(df):,}", "Total Records", "blue")
        with c2:
            render_metric_card("📐", f"{len(df.columns)}", "Data Fields", "purple")
        with c3:
            missing = df.isnull().sum().sum()
            render_metric_card("❓", f"{missing:,}", "Missing Values", "rose")
        with c4:
            trained = sum(1 for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'] if k in st.session_state)
            render_metric_card("🤖", f"{trained}/4", "Models Trained", "green")


def page_upload():
    st.markdown('<div class="step-badge">Step 1</div>', unsafe_allow_html=True)
    st.markdown("## Upload & Explore Your Data")
    st.markdown("Upload any dataset — we'll automatically figure out what each column means and prepare it for analysis.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    data_source = st.radio(
        "Choose your data source:",
        ["📂 Upload my own file", "🧪 Use demo dataset (credit card transactions)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    df = None

    if data_source == "🧪 Use demo dataset (credit card transactions)":
        if st.button("Load Demo Dataset", type="primary"):
            with st.spinner("Generating demo data with 10,000 sample transactions..."):
                df = generate_sample_data(n_samples=10000, fraud_ratio=0.02)
            st.session_state['df'] = df
            st.session_state['data_source_name'] = "Demo Dataset"
    else:
        uploaded_file = st.file_uploader(
            "Drop your CSV or Excel file here",
            type=['csv', 'xlsx', 'xls'],
            help="We support CSV and Excel files. Your data stays private and is never stored."
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state['df'] = df
                st.session_state['data_source_name'] = uploaded_file.name
            except Exception as e:
                st.error(f"We couldn't read that file. Please check it's a valid CSV or Excel file. Error: {str(e)}")
                return

    if 'df' in st.session_state:
        df = st.session_state['df']
        schema = infer_schema(df)
        st.session_state['schema'] = schema

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Data Overview")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("📋", f"{schema['total_rows']:,}", "Records", "blue")
        with c2:
            render_metric_card("📐", f"{schema['total_cols']}", "Columns", "purple")
        with c3:
            total_missing = sum(1 for v in schema['missing_pct'].values() if v > 0)
            render_metric_card("❓", f"{total_missing}", "Cols with Gaps", "rose")
        with c4:
            render_metric_card("🎯", f"{len(schema['target_candidates'])}", "Possible Targets", "green")

        st.markdown('<div class="info-box">We automatically detected your column types. Numbers, categories, dates, and potential target columns have been identified.</div>', unsafe_allow_html=True)

        with st.expander("🔍 See what we found in your data", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Number columns** (amounts, scores, measurements)")
                if schema['numeric_cols']:
                    tags = ' '.join([f'<span class="feature-tag">{friendly_feature_name(c)}</span>' for c in schema['numeric_cols']])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("*None found*")

                st.markdown("**Category columns** (types, labels, groups)")
                if schema['categorical_cols']:
                    tags = ' '.join([f'<span class="feature-tag">{friendly_feature_name(c)}</span>' for c in schema['categorical_cols']])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("*None found*")

            with col2:
                st.markdown("**Yes/No columns** (possible fraud labels)")
                if schema['binary_cols']:
                    tags = ' '.join([f'<span class="feature-tag">{friendly_feature_name(c)}</span>' for c in schema['binary_cols']])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("*None found*")

                st.markdown("**ID columns** (will be excluded from analysis)")
                if schema['id_cols']:
                    tags = ' '.join([f'<span class="feature-tag">{friendly_feature_name(c)}</span>' for c in schema['id_cols']])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("*None found*")

        target_options = ["None (find anomalies automatically)"] + list(df.columns)
        default_idx = 0
        if schema['target_candidates']:
            for tc in schema['target_candidates']:
                if tc in target_options:
                    default_idx = target_options.index(tc)
                    break

        target_column = st.selectbox(
            "Which column tells us if something is suspicious/fraudulent? (optional)",
            target_options,
            index=default_idx,
            help="If your data has a column marking fraud/normal, select it. If not, we'll find anomalies automatically."
        )

        if target_column == "None (find anomalies automatically)":
            st.session_state['target_column'] = None
            st.markdown('<div class="info-box">No target selected — we\'ll use unsupervised anomaly detection to find unusual patterns automatically.</div>', unsafe_allow_html=True)
        else:
            st.session_state['target_column'] = target_column
            vc = df[target_column].value_counts()
            st.markdown(f'<div class="info-box">Target column "{target_column}" has {len(vc)} unique values: {dict(vc.head(5))}</div>', unsafe_allow_html=True)

        st.markdown("### Data Preview")
        st.dataframe(df.head(50), use_container_width=True, height=300)

        with st.expander("📊 See detailed statistics"):
            st.dataframe(df.describe(), use_container_width=True)

            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1 and len(numeric_df.columns) <= 20:
                st.markdown("### How columns relate to each other")
                corr = numeric_df.corr()
                fig = px.imshow(
                    corr, text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Map'
                )
                fig = plotly_dark_layout(fig, 'Correlation Map')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        if target_column and target_column != "None (find anomalies automatically)":
            st.markdown("### Distribution")
            vc = df[target_column].value_counts()
            labels = [str(x) for x in vc.index]
            colors = ['#10b981', '#f43f5e'] + ['#3b82f6', '#8b5cf6', '#f59e0b'] * 10
            fig = go.Figure(data=[go.Bar(x=labels, y=vc.values, marker_color=colors[:len(labels)])])
            fig = plotly_dark_layout(fig, f'Distribution of {target_column}')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


def page_analyze():
    st.markdown('<div class="step-badge">Step 2</div>', unsafe_allow_html=True)
    st.markdown("## AI Analysis")
    st.markdown("Our AI will train multiple detection models on your data. Just pick what you want and click the button.")

    if 'df' not in st.session_state:
        st.markdown('<div class="info-box">Please upload your data first in the "Upload Data" section.</div>', unsafe_allow_html=True)
        return

    df = st.session_state['df']
    target_column = st.session_state.get('target_column', None)
    has_target = target_column is not None and target_column in df.columns

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if has_target:
        st.markdown("### What should we analyze?")
        st.markdown('<div class="info-box">Since your data has labels, we can use both supervised (learns from examples) and unsupervised (finds unusual patterns) approaches.</div>', unsafe_allow_html=True)

        analysis_tabs = st.tabs(["🎯 Smart Detection", "🔬 Anomaly Hunting", "🧠 Ensemble Power", "🌐 Deep Learning"])

        with analysis_tabs[0]:
            st.markdown("#### Smart Detection Models")
            st.markdown("These models learn from your labeled data to recognize suspicious patterns.")
            use_smote = st.checkbox(
                "Balance the dataset (recommended if fraud cases are rare)",
                value=True,
                help="If suspicious cases are much rarer than normal ones, this creates synthetic examples to help the models learn better."
            )

            if st.button("Start Smart Detection", type="primary", key="train_sup"):
                with st.spinner("Preparing your data..."):
                    result = smart_preprocess(df, target_column, st.session_state.get('schema'))
                    if result[0] is None:
                        st.error("Something went wrong with data preparation.")
                        return
                    X_train, X_test, y_train, y_test, feature_names, scaler = result
                    st.session_state['preprocessed_data'] = {
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test,
                        'feature_names': feature_names, 'scaler': scaler
                    }

                progress = st.progress(0)
                status = st.empty()

                def update_progress(msg, pct):
                    progress.progress(pct)
                    status.text(msg)

                results = train_supervised_models(X_train, X_test, y_train, y_test, use_smote, update_progress)
                st.session_state['supervised_results'] = results
                status.text("All models trained successfully!")
                st.rerun()

            if 'supervised_results' in st.session_state:
                _display_model_results(st.session_state['supervised_results'], "Supervised")

        with analysis_tabs[1]:
            st.markdown("#### Anomaly Hunting Models")
            st.markdown("These models find unusual patterns without being told what to look for — great for discovering hidden threats.")

            contamination = st.slider(
                "How much suspicious activity do you expect?",
                min_value=1, max_value=20, value=2, step=1,
                format="%d%%",
                help="Roughly what percentage of your data might be suspicious? Most datasets have 1-5% anomalies."
            )

            if st.button("Start Anomaly Hunting", type="primary", key="train_anom"):
                if 'preprocessed_data' not in st.session_state:
                    with st.spinner("Preparing your data..."):
                        result = smart_preprocess(df, target_column, st.session_state.get('schema'))
                        X_train, X_test, y_train, y_test, feature_names, scaler = result
                        st.session_state['preprocessed_data'] = {
                            'X_train': X_train, 'X_test': X_test,
                            'y_train': y_train, 'y_test': y_test,
                            'feature_names': feature_names, 'scaler': scaler
                        }

                ppd = st.session_state['preprocessed_data']
                progress = st.progress(0)
                status = st.empty()

                def update_progress(msg, pct):
                    progress.progress(pct)
                    status.text(msg)

                results = train_anomaly_models(ppd['X_train'], ppd['X_test'], ppd['y_train'], ppd['y_test'], contamination / 100, update_progress)
                st.session_state['anomaly_results'] = results
                status.text("Anomaly detection complete!")
                st.rerun()

            if 'anomaly_results' in st.session_state:
                _display_model_results(st.session_state['anomaly_results'], "Anomaly")

        with analysis_tabs[2]:
            st.markdown("#### Ensemble Power Models")
            st.markdown("Combines multiple AI models together — like getting a second and third opinion from different experts.")

            use_smote_ens = st.checkbox("Balance the dataset", value=True, key="smote_ens")

            if st.button("Start Ensemble Analysis", type="primary", key="train_ens"):
                if 'preprocessed_data' not in st.session_state:
                    with st.spinner("Preparing your data..."):
                        result = smart_preprocess(df, target_column, st.session_state.get('schema'))
                        X_train, X_test, y_train, y_test, feature_names, scaler = result
                        st.session_state['preprocessed_data'] = {
                            'X_train': X_train, 'X_test': X_test,
                            'y_train': y_train, 'y_test': y_test,
                            'feature_names': feature_names, 'scaler': scaler
                        }

                ppd = st.session_state['preprocessed_data']
                progress = st.progress(0)
                status = st.empty()

                def update_progress(msg, pct):
                    progress.progress(pct)
                    status.text(msg)

                results = train_ensemble_model(ppd['X_train'], ppd['X_test'], ppd['y_train'], ppd['y_test'], use_smote_ens, update_progress)
                st.session_state['ensemble_results'] = results
                status.text("Ensemble models trained!")
                st.rerun()

            if 'ensemble_results' in st.session_state:
                _display_model_results(st.session_state['ensemble_results'], "Ensemble")

        with analysis_tabs[3]:
            st.markdown("#### Deep Learning (Autoencoder)")
            st.markdown("A neural network that learns what 'normal' looks like, then flags anything that doesn't fit the pattern.")

            col1, col2 = st.columns(2)
            with col1:
                ae_epochs = st.slider("Training intensity", 10, 100, 30, step=10, help="More = better results but takes longer")
            with col2:
                ae_threshold = st.slider("Sensitivity", 85, 99, 95, help="Higher = fewer false alarms, but might miss some real issues")

            if st.button("Start Deep Learning", type="primary", key="train_ae"):
                if 'preprocessed_data' not in st.session_state:
                    with st.spinner("Preparing your data..."):
                        result = smart_preprocess(df, target_column, st.session_state.get('schema'))
                        X_train, X_test, y_train, y_test, feature_names, scaler = result
                        st.session_state['preprocessed_data'] = {
                            'X_train': X_train, 'X_test': X_test,
                            'y_train': y_train, 'y_test': y_test,
                            'feature_names': feature_names, 'scaler': scaler
                        }

                ppd = st.session_state['preprocessed_data']
                with st.spinner("Training deep learning model..."):
                    results = train_autoencoder(ppd['X_train'], ppd['X_test'], ppd['y_train'], ppd['y_test'],
                                               epochs=ae_epochs, threshold_percentile=ae_threshold)
                    st.session_state['autoencoder_results'] = results
                st.rerun()

            if 'autoencoder_results' in st.session_state:
                _display_model_results(st.session_state['autoencoder_results'], "Autoencoder")

    else:
        st.markdown("### Unsupervised Anomaly Detection")
        st.markdown('<div class="info-box">Since no target column was selected, we\'ll use AI to automatically find unusual patterns in your data.</div>', unsafe_allow_html=True)

        contamination = st.slider(
            "How much suspicious activity do you expect?",
            min_value=1, max_value=20, value=2, step=1,
            format="%d%%"
        )

        run_autoencoder = st.checkbox("Also run deep learning model (slower but more thorough)", value=False)

        if st.button("Find Anomalies", type="primary", key="train_unsupervised"):
            with st.spinner("Preparing your data..."):
                result = smart_preprocess(df, None, st.session_state.get('schema'))
                X_train, X_test, y_train, y_test, feature_names, scaler = result
                st.session_state['preprocessed_data'] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'feature_names': feature_names, 'scaler': scaler
                }

            progress = st.progress(0)
            status = st.empty()

            def update_progress(msg, pct):
                progress.progress(pct)
                status.text(msg)

            results = train_anomaly_models(X_train, X_test, y_train, y_test, contamination / 100, update_progress)
            st.session_state['anomaly_results'] = results
            status.text("Analysis complete!")

            if run_autoencoder:
                ae_results = train_autoencoder(X_train, X_test, y_train, y_test, epochs=30, threshold_percentile=95)
                st.session_state['autoencoder_results'] = ae_results

            st.rerun()

        if 'anomaly_results' in st.session_state:
            _display_unsupervised_results(st.session_state['anomaly_results'])

        if 'autoencoder_results' in st.session_state:
            _display_unsupervised_results(st.session_state['autoencoder_results'])


def _display_model_results(results, model_type):
    st.markdown(f"### {model_type} Results")

    has_metrics = any('accuracy' in res for res in results.values())

    if has_metrics:
        metrics_data = []
        for name, res in results.items():
            if 'accuracy' in res:
                metrics_data.append({
                    'Model': name,
                    'Overall Accuracy': res['accuracy'],
                    'Alert Quality': res['precision'],
                    'Catch Rate': res['recall'],
                    'Balance Score': res['f1'],
                    'Detection Power': res['roc_auc']
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data).set_index('Model')

            best_model = metrics_df['Balance Score'].idxmax()
            best_score = metrics_df['Balance Score'].max()

            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #10b981 !important; margin-top: 0;">Best Model: {best_model}</h4>
                <p>{explain_metric('f1', best_score)}</p>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(len(metrics_data))
            for i, (name, res) in enumerate(results.items()):
                if 'accuracy' not in res:
                    continue
                with cols[i % len(cols)]:
                    score = res['f1']
                    level, css_class, _ = get_risk_level(1 - score)
                    color = "#10b981" if score >= 0.8 else ("#f59e0b" if score >= 0.5 else "#f43f5e")
                    st.markdown(f"""
                    <div class="metric-card blue">
                        <div class="metric-value" style="color: {color} !important; font-size: 1.5rem;">{score:.1%}</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📊 Detailed Performance Breakdown"):
                for name, res in results.items():
                    if 'accuracy' not in res:
                        continue
                    st.markdown(f"**{name}**")
                    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                        st.markdown(f"- {explain_metric(metric_name, res[metric_name])}")
                    st.markdown("---")

            with st.expander("📈 Visual Comparisons"):
                model_names = list(metrics_df.index)
                metric_cols = ['Overall Accuracy', 'Alert Quality', 'Catch Rate', 'Balance Score', 'Detection Power']

                fig = go.Figure()
                colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#f43f5e']
                for i, metric in enumerate(metric_cols):
                    fig.add_trace(go.Bar(
                        name=metric, x=model_names, y=metrics_df[metric],
                        text=metrics_df[metric].apply(lambda x: f'{x:.3f}'),
                        textposition='auto',
                        marker_color=colors[i % len(colors)]
                    ))
                fig = plotly_dark_layout(fig, 'Model Performance Comparison')
                fig.update_layout(barmode='group', height=450)
                st.plotly_chart(fig, use_container_width=True)

                y_test = st.session_state['preprocessed_data'].get('y_test')
                if y_test is not None:
                    fig_roc = go.Figure()
                    for name, res in results.items():
                        if 'probabilities' in res:
                            fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
                        elif 'anomaly_scores' in res:
                            fpr, tpr, _ = roc_curve(y_test, res['anomaly_scores'])
                        else:
                            continue
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} ({res.get("roc_auc", 0):.3f})', mode='lines'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', mode='lines', line=dict(dash='dash', color='gray')))
                    fig_roc = plotly_dark_layout(fig_roc, 'Detection Performance Curves')
                    fig_roc.update_layout(height=400, xaxis_title='False Alarm Rate', yaxis_title='Detection Rate')
                    st.plotly_chart(fig_roc, use_container_width=True)

    for name, res in results.items():
        if 'anomaly_scores' in res:
            with st.expander(f"🔍 {name} - Score Distribution"):
                scores = res['anomaly_scores']
                ppd = st.session_state.get('preprocessed_data', {})
                y_test = ppd.get('y_test') if ppd else None
                if y_test is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=scores[y_test == 0], name='Normal', opacity=0.7, nbinsx=50, marker_color='#10b981'))
                    fig.add_trace(go.Histogram(x=scores[y_test == 1], name='Suspicious', opacity=0.7, nbinsx=50, marker_color='#f43f5e'))
                    fig = plotly_dark_layout(fig, f'{name} - Anomaly Scores')
                    fig.update_layout(barmode='overlay', height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    preds = res['predictions']
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=scores[preds == 0], name='Normal', opacity=0.7, nbinsx=50, marker_color='#10b981'))
                    fig.add_trace(go.Histogram(x=scores[preds == 1], name='Anomaly', opacity=0.7, nbinsx=50, marker_color='#f43f5e'))
                    fig = plotly_dark_layout(fig, f'{name} - Anomaly Scores')
                    fig.update_layout(barmode='overlay', height=350)
                    st.plotly_chart(fig, use_container_width=True)


def _display_unsupervised_results(results):
    st.markdown("### Anomaly Detection Results")

    for name, res in results.items():
        predictions = res['predictions']
        n_anomalies = int(predictions.sum())
        n_total = len(predictions)
        pct = n_anomalies / n_total * 100

        level_text, level_css, status_css = get_risk_level(pct / 100 * 5)

        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #f1f5f9 !important; margin-top: 0;">{name}</h4>
            <div class="metric-value" style="color: {'#f43f5e' if pct > 5 else '#f59e0b' if pct > 2 else '#10b981'} !important;">
                {n_anomalies:,} anomalies found ({pct:.1f}%)
            </div>
            <p>Out of {n_total:,} records checked</p>
        </div>
        """, unsafe_allow_html=True)

        if 'anomaly_scores' in res:
            scores = res['anomaly_scores']
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=scores[predictions == 0], name='Normal', opacity=0.7, nbinsx=50, marker_color='#10b981'))
            fig.add_trace(go.Histogram(x=scores[predictions == 1], name='Anomaly', opacity=0.7, nbinsx=50, marker_color='#f43f5e'))
            fig = plotly_dark_layout(fig, f'{name} - Anomaly Score Distribution')
            fig.update_layout(barmode='overlay', height=350)
            st.plotly_chart(fig, use_container_width=True)


def page_dashboard():
    st.markdown('<div class="step-badge">Intelligence Center</div>', unsafe_allow_html=True)
    st.markdown("## Model Intelligence Dashboard")

    has_any = any(k in st.session_state for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'])

    if not has_any:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px 40px;">
            <div style="font-size: 3rem; margin-bottom: 16px;">📊</div>
            <h3 style="color: #f1f5f9 !important;">No Models Trained Yet</h3>
            <p style="max-width: 400px; margin: 0 auto;">Head over to <strong>AI Analysis</strong> to train your models. Once complete, this dashboard will light up with insights.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    all_results = {}
    for key in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results']:
        if key in st.session_state:
            all_results.update(st.session_state[key])

    results_with_metrics = {k: v for k, v in all_results.items() if 'f1' in v}
    ppd = st.session_state.get('preprocessed_data', {})
    y_test = ppd.get('y_test') if ppd else None

    best_f1 = max((v['f1'] for v in results_with_metrics.values()), default=0) if results_with_metrics else 0
    best_recall = max((v['recall'] for v in results_with_metrics.values()), default=0) if results_with_metrics else 0
    best_auc = max((v['roc_auc'] for v in results_with_metrics.values()), default=0) if results_with_metrics else 0

    total_anomalies = 0
    total_checked = 0
    for res in all_results.values():
        if 'predictions' in res:
            preds = res['predictions']
            total_anomalies += int(preds.sum())
            total_checked += len(preds)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("🤖", f"{len(all_results)}", "Models", "blue")
        with c2:
            f1_pct = f"{best_f1:.0%}" if results_with_metrics else "—"
            render_metric_card("🏆", f1_pct, "Best Score", "green")
        with c3:
            recall_pct = f"{best_recall:.0%}" if results_with_metrics else "—"
            render_metric_card("🔍", recall_pct, "Catch Rate", "purple")
        with c4:
            anom_txt = f"{total_anomalies:,}" if total_checked > 0 else "—"
            render_metric_card("🚨", anom_txt, "Flagged", "rose")

    with col_right:
        if results_with_metrics:
            grade = "A+" if best_f1 >= 0.95 else "A" if best_f1 >= 0.9 else "B+" if best_f1 >= 0.85 else "B" if best_f1 >= 0.8 else "C+" if best_f1 >= 0.7 else "C" if best_f1 >= 0.6 else "D"
            grade_color = "#10b981" if best_f1 >= 0.85 else "#f59e0b" if best_f1 >= 0.7 else "#f43f5e"
            ring_bg = f"conic-gradient({grade_color} {best_f1*360}deg, rgba(255,255,255,0.05) {best_f1*360}deg)"
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 20px;">
                <div class="score-ring" style="background: {ring_bg};">
                    <div class="score-ring-inner">
                        <div class="score-ring-value" style="color: {grade_color} !important;">{grade}</div>
                        <div class="score-ring-label">Overall</div>
                    </div>
                </div>
                <p style="margin: 0; font-size: 0.85rem;">System Performance Grade</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🔬</div>
                <p>Unsupervised mode — anomaly counts shown</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if results_with_metrics:
        col_radar, col_rank = st.columns([3, 2])

        with col_radar:
            st.markdown("### Model Fingerprint")
            categories = ['Overall Accuracy', 'Alert Quality', 'Catch Rate', 'Balance Score', 'Detection Power']
            fig_radar = go.Figure()
            colors_list = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#f43f5e', '#06b6d4', '#a855f7']
            for i, (name, res) in enumerate(results_with_metrics.items()):
                vals = [res.get('accuracy', 0), res.get('precision', 0), res.get('recall', 0),
                        res.get('f1', 0), res.get('roc_auc', 0)]
                vals.append(vals[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=categories + [categories[0]], fill='toself',
                    name=name, line=dict(color=colors_list[i % len(colors_list)]),
                    fillcolor=hex_to_rgba(colors_list[i % len(colors_list)], 0.1),
                    opacity=0.8
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)',
                                    tickfont=dict(color='#64748b', size=10)),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.1)',
                                     tickfont=dict(color='#94a3b8', size=12))
                ),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9'), showlegend=True,
                legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
                height=420, margin=dict(t=30, b=30, l=60, r=60)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_rank:
            st.markdown("### Model Leaderboard")
            sorted_models = sorted(results_with_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
            for rank, (name, res) in enumerate(sorted_models, 1):
                rank_css = f"rank-{rank}" if rank <= 3 else "rank-other"
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                bar_width = res['f1'] * 100
                st.markdown(f"""
                <div class="model-rank">
                    <div class="rank-badge {rank_css}">{medal}</div>
                    <div class="model-rank-name">{name}</div>
                    <div class="model-rank-score">{res['f1']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            insights = []
            if best_f1 >= 0.95:
                insights.append(('<span class="insight-pill insight-good">Excellent detection</span>', ""))
            elif best_f1 >= 0.8:
                insights.append(('<span class="insight-pill insight-good">Strong detection</span>', ""))
            else:
                insights.append(('<span class="insight-pill insight-warn">Room to improve</span>', ""))

            if best_recall >= 0.9:
                insights.append(('<span class="insight-pill insight-good">Low miss rate</span>', ""))
            elif best_recall < 0.7:
                insights.append(('<span class="insight-pill insight-bad">High miss rate</span>', ""))

            f1_values = [v['f1'] for v in results_with_metrics.values()]
            if len(f1_values) > 1 and max(f1_values) - min(f1_values) < 0.05:
                insights.append(('<span class="insight-pill insight-warn">Models perform similarly</span>', ""))
            elif len(f1_values) > 1:
                insights.append(('<span class="insight-pill insight-good">Clear winner exists</span>', ""))

            st.markdown("<div style='margin-top: 12px;'>" + " ".join([i[0] for i in insights]) + "</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_cm, col_roc = st.columns(2)

        with col_cm:
            best_model_name = max(results_with_metrics, key=lambda x: results_with_metrics[x]['f1'])
            best_res = results_with_metrics[best_model_name]
            if 'predictions' in best_res and y_test is not None:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(y_test, best_res['predictions'])
                labels = ['Normal', 'Anomaly']
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm, x=labels, y=labels,
                    text=[[f"{v:,}" for v in row] for row in cm],
                    texttemplate="%{text}", textfont=dict(size=18, color='white'),
                    colorscale=[[0, '#1e293b'], [0.5, '#3b82f6'], [1, '#8b5cf6']],
                    showscale=False, hoverinfo='skip'
                ))
                fig_cm.update_layout(
                    xaxis=dict(title='Predicted', tickfont=dict(color='#94a3b8'), titlefont=dict(color='#94a3b8')),
                    yaxis=dict(title='Actual', tickfont=dict(color='#94a3b8'), titlefont=dict(color='#94a3b8'), autorange='reversed'),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=380, margin=dict(t=20, b=40, l=60, r=20),
                    annotations=[dict(text=f"Best: {best_model_name}", x=0.5, y=1.08, xref='paper', yref='paper',
                                      showarrow=False, font=dict(color='#64748b', size=11))]
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                st.markdown(f"""
                <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;">
                    <span class="insight-pill insight-good">True catches: {tp:,}</span>
                    <span class="insight-pill insight-good">True clears: {tn:,}</span>
                    <span class="insight-pill insight-bad">Missed: {fn:,}</span>
                    <span class="insight-pill insight-warn">False alarms: {fp:,}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_roc:
            if y_test is not None:
                st.markdown("### ROC Curves")
                fig_roc = go.Figure()
                for i, (name, res) in enumerate(results_with_metrics.items()):
                    if 'probabilities' in res:
                        fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
                    elif 'anomaly_scores' in res:
                        fpr, tpr, _ = roc_curve(y_test, res['anomaly_scores'])
                    else:
                        continue
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, name=f'{name} ({res["roc_auc"]:.3f})',
                        mode='lines', line=dict(color=colors_list[i % len(colors_list)], width=2.5)
                    ))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess', mode='lines',
                                             line=dict(dash='dot', color='rgba(255,255,255,0.15)', width=1)))
                fig_roc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title='False Positive Rate', gridcolor='rgba(255,255,255,0.05)',
                               tickfont=dict(color='#64748b'), titlefont=dict(color='#94a3b8')),
                    yaxis=dict(title='True Positive Rate', gridcolor='rgba(255,255,255,0.05)',
                               tickfont=dict(color='#64748b'), titlefont=dict(color='#94a3b8')),
                    legend=dict(font=dict(color='#94a3b8', size=10), bgcolor='rgba(0,0,0,0)'),
                    height=380, margin=dict(t=20, b=40, l=60, r=20),
                    font=dict(color='#f1f5f9')
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.markdown("### Anomaly Distribution")
                all_anomaly_pcts = []
                for name, res in all_results.items():
                    if 'predictions' in res:
                        pct = res['predictions'].mean() * 100
                        all_anomaly_pcts.append((name, pct))
                if all_anomaly_pcts:
                    names, pcts = zip(*all_anomaly_pcts)
                    fig_bar = go.Figure(go.Bar(
                        x=list(pcts), y=list(names), orientation='h',
                        marker=dict(color=[('#f43f5e' if p > 5 else '#f59e0b' if p > 2 else '#10b981') for p in pcts]),
                        text=[f'{p:.1f}%' for p in pcts], textposition='auto',
                        textfont=dict(color='white')
                    ))
                    fig_bar = plotly_dark_layout(fig_bar, '')
                    fig_bar.update_layout(height=380, xaxis_title='Anomaly Rate (%)')
                    st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if 'supervised_results' in st.session_state:
        feature_names = ppd.get('feature_names', [])
        for name, res in st.session_state['supervised_results'].items():
            if hasattr(res['model'], 'feature_importances_'):
                st.markdown("### Feature Influence Map")
                friendly_names = [friendly_feature_name(f) for f in feature_names]
                importance_df = pd.DataFrame({
                    'Feature': friendly_names,
                    'Importance': res['model'].feature_importances_
                }).sort_values('Importance', ascending=False)

                top_n = min(12, len(importance_df))
                top_features = importance_df.head(top_n).sort_values('Importance', ascending=True)

                max_imp = top_features['Importance'].max()
                colors_imp = []
                for v in top_features['Importance']:
                    ratio = v / max_imp if max_imp > 0 else 0
                    if ratio > 0.7:
                        colors_imp.append('#3b82f6')
                    elif ratio > 0.4:
                        colors_imp.append('#8b5cf6')
                    else:
                        colors_imp.append('#6366f1')

                fig_imp = go.Figure(go.Bar(
                    x=top_features['Importance'], y=top_features['Feature'], orientation='h',
                    marker=dict(color=colors_imp, line=dict(width=0)),
                    text=[f'{v:.4f}' for v in top_features['Importance']], textposition='outside',
                    textfont=dict(color='#94a3b8', size=11)
                ))
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#64748b'),
                               title='How much this factor matters', titlefont=dict(color='#94a3b8')),
                    yaxis=dict(tickfont=dict(color='#e2e8f0', size=12)),
                    height=max(300, top_n * 35), margin=dict(t=10, b=40, l=10, r=80),
                    font=dict(color='#f1f5f9')
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                top3 = importance_df.head(3)['Feature'].tolist()
                st.markdown(f"""
                <div class="glass-card">
                    <h4 style="color: #f1f5f9 !important; margin-top: 0;">What This Means</h4>
                    <p>The AI pays the most attention to <strong>{top3[0]}</strong>, <strong>{top3[1] if len(top3) > 1 else 'N/A'}</strong>,
                    and <strong>{top3[2] if len(top3) > 2 else 'N/A'}</strong> when deciding if something looks suspicious. 
                    If you want to reduce false alarms, focus on improving data quality for these factors.</p>
                </div>
                """, unsafe_allow_html=True)
                break

    if not results_with_metrics and total_checked > 0:
        st.markdown("### Detection Summary")
        anomaly_rate = total_anomalies / total_checked * 100 if total_checked > 0 else 0
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 800; color: {'#f43f5e' if anomaly_rate > 5 else '#f59e0b' if anomaly_rate > 2 else '#10b981'} !important;">
                {anomaly_rate:.1f}%
            </div>
            <p style="font-size: 1.1rem; margin-top: 8px;">Overall Anomaly Rate</p>
            <p>{total_anomalies:,} anomalies found in {total_checked:,} records across {len(all_results)} models</p>
        </div>
        """, unsafe_allow_html=True)

        for name, res in all_results.items():
            if 'anomaly_scores' in res:
                preds = res['predictions']
                scores = res['anomaly_scores']
                fig_scatter = go.Figure()
                indices = np.arange(len(scores))
                normal_mask = preds == 0
                anomaly_mask = preds == 1
                fig_scatter.add_trace(go.Scattergl(
                    x=indices[normal_mask], y=scores[normal_mask], mode='markers',
                    marker=dict(color='#10b981', size=3, opacity=0.4), name='Normal'
                ))
                fig_scatter.add_trace(go.Scattergl(
                    x=indices[anomaly_mask], y=scores[anomaly_mask], mode='markers',
                    marker=dict(color='#f43f5e', size=6, opacity=0.9), name='Anomaly'
                ))
                fig_scatter = plotly_dark_layout(fig_scatter, f'{name} — Anomaly Map')
                fig_scatter.update_layout(height=350, xaxis_title='Record Index', yaxis_title='Anomaly Score')
                st.plotly_chart(fig_scatter, use_container_width=True)
                break


def _run_quick_check(input_scaled, all_results_keys=None):
    all_predictions = []
    all_scores = []

    if 'supervised_results' in st.session_state:
        for name, res in st.session_state['supervised_results'].items():
            model = res['model']
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            fraud_prob = proba[1] if len(proba) > 1 else float(prediction)
            all_predictions.append({'name': name, 'prediction': prediction, 'score': fraud_prob, 'type': 'Supervised'})
            all_scores.append(fraud_prob)

    if 'anomaly_results' in st.session_state:
        for name, res in st.session_state['anomaly_results'].items():
            model = res['model']
            prediction = model.predict(input_scaled)[0]
            pred_binary = 1 if prediction == -1 else 0
            if hasattr(model, 'decision_function'):
                score = -model.decision_function(input_scaled)[0]
            else:
                score = -model.score_samples(input_scaled)[0]
            all_scores_range = res.get('anomaly_scores', np.array([score]))
            normalized_score = min(1.0, max(0.0, (score - all_scores_range.min()) / (all_scores_range.max() - all_scores_range.min() + 1e-10)))
            all_predictions.append({'name': name, 'prediction': pred_binary, 'score': normalized_score, 'type': 'Unsupervised'})
            all_scores.append(normalized_score)

    if 'ensemble_results' in st.session_state:
        for name, res in st.session_state['ensemble_results'].items():
            model = res['model']
            if 'Hybrid' in name and 'iso_forest' in res:
                iso_score = res['iso_forest'].decision_function(input_scaled).reshape(1, -1)
                inp = np.hstack([input_scaled, iso_score])
                prediction = model.predict(inp)[0]
                proba = model.predict_proba(inp)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            else:
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
            fraud_prob = proba[1] if len(proba) > 1 else float(prediction)
            all_predictions.append({'name': name, 'prediction': prediction, 'score': fraud_prob, 'type': 'Ensemble'})
            all_scores.append(fraud_prob)

    if 'autoencoder_results' in st.session_state:
        ae_res = st.session_state['autoencoder_results']['Autoencoder']
        ae_model = ae_res['model']
        reconstruction = ae_model.predict(input_scaled, verbose=0)
        mse = np.mean(np.power(input_scaled - reconstruction, 2))
        threshold = ae_res['threshold']
        is_anomaly = mse > threshold
        normalized_score = min(1.0, mse / (threshold * 2))
        all_predictions.append({'name': 'Autoencoder', 'prediction': int(is_anomaly), 'score': normalized_score, 'type': 'Deep Learning'})
        all_scores.append(normalized_score)

    return all_predictions, all_scores


def _render_quick_check_results(all_predictions, all_scores):
    if not all_scores:
        st.warning("No model predictions available.")
        return

    avg_score = np.mean(all_scores)
    n_flagged = sum(1 for p in all_predictions if p['prediction'] == 1)
    n_total = len(all_predictions)
    consensus_pct = n_flagged / n_total * 100

    level_text, level_css, status_css = get_risk_level(avg_score)

    gauge_color = "#10b981" if avg_score < 0.4 else "#f59e0b" if avg_score < 0.7 else "#f43f5e"
    gauge_angle = avg_score * 180

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score * 100,
        number=dict(suffix="%", font=dict(size=48, color=gauge_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color='#64748b'), dtick=20),
            bar=dict(color=gauge_color, thickness=0.3),
            bgcolor='rgba(255,255,255,0.03)',
            bordercolor='rgba(255,255,255,0.1)',
            steps=[
                dict(range=[0, 40], color='rgba(16,185,129,0.1)'),
                dict(range=[40, 70], color='rgba(245,158,11,0.1)'),
                dict(range=[70, 100], color='rgba(244,63,94,0.1)')
            ],
            threshold=dict(line=dict(color=gauge_color, width=3), thickness=0.8, value=avg_score * 100)
        )
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f1f5f9'),
        height=220, margin=dict(t=30, b=10, l=40, r=40)
    )

    col_gauge, col_verdict = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_verdict:
        if avg_score >= 0.7:
            verdict_css = "verdict-danger"
            verdict_icon = "🚨"
            verdict_title = "HIGH RISK"
            verdict_msg = f"{n_flagged} out of {n_total} AI models flagged this as suspicious"
        elif avg_score >= 0.4:
            verdict_css = "verdict-warning"
            verdict_icon = "⚡"
            verdict_title = "MODERATE RISK"
            verdict_msg = f"{n_flagged} out of {n_total} models found unusual patterns"
        else:
            verdict_css = "verdict-safe"
            verdict_icon = "✅"
            verdict_title = "LOW RISK"
            verdict_msg = f"{n_total - n_flagged} out of {n_total} models say this looks normal"

        st.markdown(f"""
        <div class="verdict-card {verdict_css}" style="padding: 30px;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">{verdict_icon}</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {gauge_color} !important; margin-bottom: 4px;">{verdict_title}</div>
            <div style="font-size: 1.1rem; color: #f1f5f9 !important; font-weight: 600;">{level_text}</div>
            <p style="margin-top: 12px; font-size: 0.9rem;">{verdict_msg}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Model-by-Model Verdict")

    for pred in sorted(all_predictions, key=lambda x: x['score'], reverse=True):
        dot_css = "verdict-dot-danger" if pred['prediction'] == 1 else "verdict-dot-safe"
        status_text = "FLAGGED" if pred['prediction'] == 1 else "CLEAR"
        status_color = "#f43f5e" if pred['prediction'] == 1 else "#10b981"
        bar_pct = pred['score'] * 100

        st.markdown(f"""
        <div class="model-verdict-row">
            <div class="verdict-dot {dot_css}"></div>
            <div style="flex: 1;">
                <div style="color: #f1f5f9 !important; font-weight: 600; font-size: 0.95rem;">{pred['name']}</div>
                <div style="color: #64748b !important; font-size: 0.75rem;">{pred['type']}</div>
            </div>
            <div style="width: 120px; margin-right: 12px;">
                <div style="height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden;">
                    <div style="height: 100%; width: {bar_pct}%; background: {status_color}; border-radius: 3px;"></div>
                </div>
            </div>
            <div style="min-width: 60px; text-align: right;">
                <span style="color: {status_color} !important; font-weight: 700; font-size: 0.9rem;">{status_text}</span>
            </div>
            <div style="min-width: 50px; text-align: right; color: #94a3b8 !important; font-size: 0.85rem;">{pred['score']:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    if n_total > 1:
        agreement = max(n_flagged, n_total - n_flagged) / n_total * 100
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 16px; text-align: center;">
            <p style="margin: 0;"><strong style="color: #f1f5f9 !important;">Model Agreement: {agreement:.0f}%</strong> — 
            {'All models agree on this verdict' if agreement == 100 else 'Most models agree' if agreement >= 75 else 'Models are split — review manually'}</p>
        </div>
        """, unsafe_allow_html=True)


def page_quick_check():
    st.markdown('<div class="step-badge">Live Testing</div>', unsafe_allow_html=True)
    st.markdown("## Check a Record")

    has_trained = any(k in st.session_state for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'])
    has_loaded = 'loaded_models' in st.session_state

    if not has_trained and not has_loaded:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 40px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🚀</div>
            <h3 style="color: #818cf8 !important; margin-bottom: 8px;">Welcome to Quick Check</h3>
            <p style="max-width: 520px; margin: 0 auto; color: #94a3b8 !important; font-size: 0.95rem; line-height: 1.7;">
                This is the fastest way to try the AI. Click the button below and we'll:
            </p>
            <div style="text-align: left; max-width: 400px; margin: 16px auto; color: #c8d6e5; font-size: 0.9rem; line-height: 2.2;">
                <strong>1.</strong> Load a sample dataset of 10,000 credit card transactions<br>
                <strong>2.</strong> Train 3 AI models to detect fraud automatically<br>
                <strong>3.</strong> Bring you right back here to start testing
            </div>
            <p style="color: #64748b !important; font-size: 0.82rem; margin-top: 8px;">
                The whole process takes about 30 seconds. No data files or setup needed.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Set Up AI & Start Checking", type="primary", key="quick_start_auto", use_container_width=True):
            with st.spinner("Step 1 of 3 — Loading sample transactions..."):
                df = generate_sample_data(n_samples=10000, fraud_ratio=0.02)
                st.session_state['df'] = df
                st.session_state['data_source_name'] = "Demo Dataset (Quick Start)"
                schema = infer_schema(df)
                st.session_state['schema'] = schema
                target_column = 'Is_Fraud'
                st.session_state['target_column'] = target_column

            with st.spinner("Step 2 of 3 — Training AI models on the data..."):
                result = smart_preprocess(df, target_column, schema)
                if result[0] is None:
                    st.error("Something went wrong preparing the data. Please try again.")
                    return
                X_train, X_test, y_train, y_test, feature_names, scaler = result
                st.session_state['preprocessed_data'] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'feature_names': feature_names, 'scaler': scaler
                }

            with st.spinner("Step 3 of 3 — Training detection models (this takes ~20 seconds)..."):
                results = train_supervised_models(X_train, X_test, y_train, y_test, use_smote=True)
                st.session_state['supervised_results'] = results

            st.success("All set! Your AI models are trained and ready. The form is below.")
            st.rerun()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="padding: 18px 24px;">
            <div style="color: #64748b !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">Already have your own data?</div>
            <p style="color: #94a3b8 !important; font-size: 0.88rem; line-height: 1.6; margin: 0;">
                If you have a CSV or Excel file you'd like to analyze, use the <strong>Full Analysis</strong> path instead. 
                Go to <strong>Upload Data</strong> in the sidebar to upload your file, then <strong>AI Analysis</strong> to train models with your own settings.
                Once trained, come back here to test individual records.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if has_trained:
        feature_names = st.session_state['preprocessed_data']['feature_names']
        scaler = st.session_state['preprocessed_data']['scaler']
    elif has_loaded:
        feature_names = st.session_state['loaded_models']['feature_names']
        scaler = st.session_state['loaded_models']['scaler']

    feature_names = list(feature_names)
    n_features = len(feature_names)

    if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
        means = scaler.mean_
        stds = np.maximum(scaler.scale_, 1e-6)
    else:
        means = np.zeros(n_features)
        stds = np.ones(n_features)

    st.markdown("""
    <div class="glass-card">
        <p style="margin: 0; color: #94a3b8 !important;">Fill in the details below to check if something looks suspicious. 
        You can type in real values, or use the quick-fill buttons to generate test scenarios. 
        All your trained AI models will evaluate it at once.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("#### Quick-Fill Scenarios")
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-top:-8px;">Don\'t have specific values? Generate a test scenario to see how the AI responds.</p>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        random_normal = st.button("✅ Typical / Normal", key="qc_rand_normal", use_container_width=True, help="Generate values that look like a normal, everyday record")
    with sc2:
        random_suspicious = st.button("⚡ Looks Suspicious", key="qc_rand_suspicious", use_container_width=True, help="Generate values with some unusual patterns")
    with sc3:
        random_extreme = st.button("🚨 Very Unusual", key="qc_rand_extreme", use_container_width=True, help="Generate values that are way outside the normal range")
    with sc4:
        random_edge = st.button("🔀 Mixed Signals", key="qc_rand_edge", use_container_width=True, help="Generate values where some things look normal and others don't")

    if 'qc_values' not in st.session_state:
        st.session_state['qc_values'] = {f: float(means[i]) for i, f in enumerate(feature_names)}

    scenario_label = None
    if random_normal:
        noise = np.random.normal(0, 0.3, n_features)
        st.session_state['qc_values'] = {f: float(means[i] + stds[i] * noise[i]) for i, f in enumerate(feature_names)}
        scenario_label = "typical_normal"
        st.rerun()
    elif random_suspicious:
        shifts = np.random.uniform(2, 4, n_features) * np.random.choice([-1, 1], n_features)
        st.session_state['qc_values'] = {f: float(means[i] + stds[i] * shifts[i]) for i, f in enumerate(feature_names)}
        scenario_label = "suspicious"
        st.rerun()
    elif random_extreme:
        shifts = np.random.uniform(5, 10, n_features) * np.random.choice([-1, 1], n_features)
        st.session_state['qc_values'] = {f: float(means[i] + stds[i] * shifts[i]) for i, f in enumerate(feature_names)}
        scenario_label = "extreme"
        st.rerun()
    elif random_edge:
        vals = {}
        for i, f in enumerate(feature_names):
            if np.random.random() < 0.3:
                vals[f] = float(means[i] + stds[i] * np.random.uniform(3, 6) * np.random.choice([-1, 1]))
            else:
                vals[f] = float(means[i] + stds[i] * np.random.normal(0, 0.5))
        st.session_state['qc_values'] = vals
        scenario_label = "edge"
        st.rerun()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### Enter Details")
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-top:-8px;">Fill in what you know about the record you want to check. Use the sliders and fields below.</p>', unsafe_allow_html=True)

    widget_configs = {}
    for i, f in enumerate(feature_names):
        widget_configs[f] = get_feature_widget_config(f, means[i], stds[i])

    input_data = {}
    slider_features = [f for f in feature_names if widget_configs[f]['type'] in ('risk_slider', 'trust_slider', 'pattern_slider', 'score_slider')]
    non_slider_features = [f for f in feature_names if f not in slider_features]

    if non_slider_features:
        cols_per_row = min(3, len(non_slider_features))
        for i in range(0, len(non_slider_features), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(non_slider_features):
                    feature = non_slider_features[idx]
                    with cols[j]:
                        current_val = st.session_state['qc_values'].get(feature, means[feature_names.index(feature)])
                        input_data[feature] = render_smart_input(feature, widget_configs[feature], current_val)

    if slider_features:
        st.markdown("")
        for feature in slider_features:
            current_val = st.session_state['qc_values'].get(feature, means[feature_names.index(feature)])
            input_data[feature] = render_smart_input(feature, widget_configs[feature], current_val)

    st.session_state['qc_values'] = {f: input_data.get(f, st.session_state['qc_values'].get(f, 0.0)) for f in feature_names}
    input_data = st.session_state['qc_values']

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    summary_parts = []
    for f in feature_names:
        val = input_data[f]
        config = widget_configs[f]
        wtype = config['type']
        friendly = config['label']
        if wtype == 'toggle':
            summary_parts.append(f"<strong>{friendly}</strong>: {'Yes' if val > 0.5 else 'No'}")
        elif wtype == 'dollar':
            summary_parts.append(f"<strong>{friendly}</strong>: ${val:,.2f}")
        elif wtype == 'hour':
            h = int(val)
            am_pm = "AM" if h < 12 else "PM"
            display_h = h if h <= 12 else h - 12
            if display_h == 0:
                display_h = 12
            summary_parts.append(f"<strong>{friendly}</strong>: {display_h}:00 {am_pm}")
        elif wtype == 'day':
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            summary_parts.append(f"<strong>{friendly}</strong>: {days[int(val) % 7]}")
        elif wtype == 'month':
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            summary_parts.append(f"<strong>{friendly}</strong>: {months[int(val - 1) % 12]}")
        elif wtype == 'count':
            summary_parts.append(f"<strong>{friendly}</strong>: {val:.0f} times")
        elif 'slider' in wtype:
            feat_idx = feature_names.index(f)
            z = (val - means[feat_idx]) / (stds[feat_idx] + 1e-10)
            if z > 2:
                level = "Very High"
            elif z > 1:
                level = "High"
            elif z > -1:
                level = "Normal"
            elif z > -2:
                level = "Low"
            else:
                level = "Very Low"
            summary_parts.append(f"<strong>{friendly}</strong>: {level}")
        else:
            summary_parts.append(f"<strong>{friendly}</strong>: {val:.2f}")

    st.markdown(f"""
    <div class="glass-card" style="padding: 18px 24px;">
        <div style="color: #64748b !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">What You're Checking</div>
        <div style="color: #c8d6e5 !important; font-size: 0.9rem; line-height: 1.8;">{'&nbsp;&nbsp;·&nbsp;&nbsp;'.join(summary_parts)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if st.button("⚡ Run AI Check", type="primary", key="quick_check_btn", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        all_predictions, all_scores = _run_quick_check(input_scaled)
        _render_quick_check_results(all_predictions, all_scores)

    with st.expander("🔧 Advanced: View Raw Values", expanded=False):
        raw_cols = st.columns(min(4, n_features))
        for i, f in enumerate(feature_names):
            with raw_cols[i % min(4, n_features)]:
                st.markdown(f"**{friendly_feature_name(f)}**")
                st.code(f"{input_data[f]:.6f}")


def page_reports():
    st.markdown('<div class="step-badge">Export</div>', unsafe_allow_html=True)
    st.markdown("## Reports & Saved Models")

    has_any = any(k in st.session_state for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'])

    if has_any:
        st.markdown("### Download Reports")
        all_results = {}
        for key in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results']:
            if key in st.session_state:
                all_results.update(st.session_state[key])

        col1, col2 = st.columns(2)
        with col1:
            html_report = generate_report_html(all_results)
            st.download_button(
                label="📄 Download Full Report (HTML)",
                data=html_report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
        with col2:
            results_with_metrics = {k: v for k, v in all_results.items() if 'accuracy' in v}
            if results_with_metrics:
                friendly_metric_names = {
                    'accuracy': 'Overall Accuracy', 'precision': 'Alert Quality',
                    'recall': 'Catch Rate', 'f1': 'Balance Score', 'roc_auc': 'Detection Power'
                }
                csv_data = pd.DataFrame({
                    name: {friendly_metric_names.get(m, m): res[m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] if m in res}
                    for name, res in results_with_metrics.items()
                }).T
                csv_data.index.name = 'Model'
                st.download_button(
                    label="📊 Download Metrics (CSV)",
                    data=csv_data.to_csv(),
                    file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("### Save Models")
        model_types = []
        if 'supervised_results' in st.session_state:
            model_types.append("Smart Detection")
        if 'anomaly_results' in st.session_state:
            model_types.append("Anomaly Detection")
        if 'ensemble_results' in st.session_state:
            model_types.append("Ensemble")
        if 'autoencoder_results' in st.session_state:
            model_types.append("Deep Learning")

        save_type = st.selectbox("Which models to save?", model_types)

        if st.button("💾 Save Models", type="primary", key="save_models"):
            result_key = {
                "Smart Detection": "supervised_results",
                "Anomaly Detection": "anomaly_results",
                "Ensemble": "ensemble_results",
                "Deep Learning": "autoencoder_results"
            }[save_type]

            results = st.session_state[result_key]
            scaler = st.session_state['preprocessed_data']['scaler']
            feature_names = st.session_state['preprocessed_data']['feature_names']
            save_dir = save_models(save_type, results, scaler, feature_names)
            st.success(f"Models saved successfully!")

    else:
        st.markdown('<div class="info-box">Train models first to generate reports and save them.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Previously Saved Models")
    saved_models = load_saved_models()

    if saved_models:
        for idx, meta in enumerate(saved_models):
            with st.expander(f"📦 {meta['model_type']} — Saved {meta['timestamp']} ({len(meta['models'])} models)"):
                st.write(f"**Models:** {', '.join(meta['models'])}")
                st.write(f"**Features used:** {meta['n_features']}")

                if meta['metrics']:
                    metrics_df = pd.DataFrame(meta['metrics']).T
                    st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.4f}"), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Load These Models", key=f"load_{idx}"):
                        try:
                            models, scaler, feature_names, extras = load_model_from_dir(meta['dir_path'], meta)
                            st.session_state['loaded_models'] = {
                                'models': models, 'scaler': scaler,
                                'feature_names': feature_names,
                                'metadata': meta, 'extras': extras
                            }
                            st.success(f"Loaded {len(models)} models!")
                        except Exception as e:
                            st.error(f"Error loading: {str(e)}")
                with col2:
                    if st.button(f"Delete", key=f"delete_{idx}"):
                        import shutil
                        shutil.rmtree(meta['dir_path'])
                        st.success("Deleted!")
                        st.rerun()
    else:
        st.markdown("*No saved models found yet.*")


def main():
    inject_custom_css()

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 16px 0;">
            <div style="font-size: 2rem;">🛡️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9; margin-top: 4px;">AI Detector</div>
            <div style="font-size: 0.75rem; color: #64748b;">Anomaly & Fraud Detection</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        nav_options = ["🏠 Home", "📂 Upload Data", "🤖 AI Analysis", "📊 Dashboard", "🔍 Quick Check", "📄 Reports & Models"]
        nav_target = st.session_state.pop('nav_target', None)
        default_index = nav_options.index(nav_target) if nav_target and nav_target in nav_options else 0

        page = st.radio(
            "Navigation",
            nav_options,
            index=default_index,
            label_visibility="collapsed"
        )

        st.markdown("---")

        if 'df' in st.session_state:
            df = st.session_state['df']
            source = st.session_state.get('data_source_name', 'Unknown')
            st.markdown(f"**Loaded:** {source}")
            st.markdown(f"**Records:** {len(df):,}")

            trained = []
            if 'supervised_results' in st.session_state:
                trained.append("Smart")
            if 'anomaly_results' in st.session_state:
                trained.append("Anomaly")
            if 'ensemble_results' in st.session_state:
                trained.append("Ensemble")
            if 'autoencoder_results' in st.session_state:
                trained.append("Deep")
            if trained:
                st.markdown(f"**Trained:** {', '.join(trained)}")

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.7rem; color: #475569; padding: 8px;">
            AI-Powered Analysis<br>
            Works with any dataset
        </div>
        """, unsafe_allow_html=True)

    if page == "🏠 Home":
        page_home()
    elif page == "📂 Upload Data":
        page_upload()
    elif page == "🤖 AI Analysis":
        page_analyze()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "🔍 Quick Check":
        page_quick_check()
    elif page == "📄 Reports & Models":
        page_reports()


if __name__ == "__main__":
    main()
