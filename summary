# AI Anomaly & Fraud Detector

## Overview
A modern Streamlit-based AI platform for anomaly detection and fraud analysis. Works with any dataset — users upload CSV/Excel files and the AI automatically detects patterns, anomalies, and suspicious activity. Designed to be user-friendly for non-technical users with plain English explanations. Built as a portfolio piece for data science / anomaly detection roles.

## Project Architecture
- **app.py** - Main Streamlit application (~2500 lines) with:
  - Custom dark glassmorphism UI theme with CSS animations
  - Smart schema inference (auto-detects column types)
  - Universal preprocessing pipeline (handles numeric, categorical, datetime, mixed data)
  - Supervised models (Logistic Regression, Random Forest, XGBoost)
  - Anomaly detection (Isolation Forest, One-Class SVM, LOF)
  - Ensemble models (Voting, Stacking, Hybrid)
  - Autoencoder (TensorFlow/Keras deep learning)
  - Model Intelligence Dashboard with radar charts, confusion matrix, ROC curves, leaderboard, feature influence map
  - Quick Check (Live Testing) with random scenario generators and gauge-based risk visualization
  - Report generation and model save/load
- **.streamlit/config.toml** - Streamlit config (dark theme, port 5000)
- **saved_models/** - Directory for persisted trained models
- **requirements.txt** - Python dependencies
- **pyproject.toml** - Build system config

## Tech Stack
- **Language**: Python 3.11
- **Frontend**: Streamlit (port 5000, dark theme)
- **ML Libraries**: scikit-learn, XGBoost, TensorFlow, imbalanced-learn
- **Visualization**: Plotly (dark-themed charts - radar, gauge, heatmap, scatter)
- **Data**: Pandas, NumPy, SciPy

## Running
The app runs via `streamlit run app.py` on port 5000 with address 0.0.0.0.

## Pages (Sidebar Navigation)
1. **Home** - Landing page with two user paths: Quick Check (non-technical, one-click setup) and Full Analysis (technical, upload own data), comparison table, feature cards
2. **Upload Data** - Upload CSV/Excel or use demo dataset, auto-schema detection
3. **AI Analysis** - Train models with plain English explanations
4. **Dashboard** - Intelligence center: radar chart model fingerprints, performance grade ring, leaderboard, confusion matrix heatmap, ROC curves, feature influence map, anomaly scatter plots
5. **Quick Check** - One-click Quick Start (auto-loads demo data + trains models), smart form with context-aware inputs (dollar amounts, time pickers, risk sliders, frequency counters), quick-fill scenario generators, "What You're Checking" plain English summary, gauge risk meter, model-by-model verdict cards
6. **Reports & Models** - Download HTML/CSV reports, save/load models

## Key Features
- Works with ANY dataset (not just credit card fraud)
- Auto-detects column types (numeric, categorical, datetime, binary, ID)
- Handles datetime feature extraction (hour, day of week, month)
- Unsupervised mode when no target column exists
- Plain English metric explanations
- Quick Check with random test scenario generators
- Radar chart model fingerprints and confusion matrix heatmaps
- Risk score consensus from multiple AI models
- Modern dark UI with glassmorphism design

## Helper Functions
- `hex_to_rgba(hex_color, alpha)` - Converts hex colors to rgba for Plotly chart transparency
- `get_risk_level(score)` - Returns risk text, CSS class based on score threshold
- `render_metric_card(icon, value, label, color)` - Renders styled metric cards
- `plotly_dark_layout(fig, title)` - Applies consistent dark theme to Plotly charts
- `friendly_feature_name(raw_name)` - Converts V1/snake_case/camelCase to friendly display names
- `get_feature_widget_config(name, mean, std)` - Auto-detects feature type and returns smart widget config (dollar, time, slider, count, toggle)
- `render_smart_input(feature, config, val)` - Renders context-appropriate Streamlit widget based on config
- `is_dummy_feature(name)` - Detects one-hot/dummy encoded categorical features

## Recent Changes
- 2026-02-06: Two user paths - Home page redesigned with Quick Check (non-technical) and Full Analysis (technical) paths, comparison table, navigation buttons; Quick Check page gets one-click Quick Start that auto-loads demo data and trains models
- 2026-02-06: Quick Check smart form redesign - context-aware inputs (dollar for amounts, time pickers, risk/trust sliders with Low/High labels, frequency counters, Yes/No toggles for dummy features), scenario generators use scaler stats, plain English "What You're Checking" summary, raw values in advanced expander
- 2026-02-06: Dashboard redesign - radar chart model fingerprints, performance grade ring, leaderboard with medals, confusion matrix heatmap, ROC curves, feature influence map with key insights, anomaly scatter plots
- 2026-02-06: Quick Check redesign - random scenario generators (normal/suspicious/extreme/edge case), gauge risk meter, model-by-model verdict cards, model agreement indicator
- 2026-02-06: Complete redesign - dark glassmorphism UI, sidebar navigation, smart schema inference, universal preprocessing, Quick Check page, plain English explanations, unsupervised mode support
- 2026-02-06: Initial Replit setup - configured Streamlit for port 5000, installed dependencies
