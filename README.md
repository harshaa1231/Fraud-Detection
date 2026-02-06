# AI Anomaly & Fraud Detector

A production-grade, AI-powered anomaly detection platform built with Python and Streamlit. Upload any dataset and let 10+ machine learning models automatically find hidden patterns, anomalies, and suspicious activity — all explained in plain English.

Built as a portfolio project demonstrating expertise in **anomaly detection**, **fraud detection**, **machine learning engineering**, and **data science**.

---

## Live Demo

The app runs on **Streamlit** with a modern dark glassmorphism UI. Two user paths are available from the home page:

| | Quick Check | Full Analysis |
|---|---|---|
| **For** | Non-technical users | Data scientists & analysts |
| **Data** | Sample dataset provided | Upload your own CSV/Excel |
| **Setup** | One click — fully automatic | Choose models & settings |
| **Results** | Simple risk score & verdict | Charts, metrics & reports |
| **Time** | ~1 minute | 5-15 minutes |

---

## Features

### Universal Dataset Support
- Works with **any** CSV or Excel dataset — not limited to credit card fraud
- Smart schema inference automatically detects column types (numeric, categorical, datetime, binary, ID)
- Universal preprocessing pipeline handles mixed data formats, missing values, and feature engineering
- Datetime feature extraction (hour of day, day of week, month)

### 10+ AI Models Across 4 Categories

**Supervised Detection** — Learns from labeled examples
- Logistic Regression
- Random Forest
- XGBoost (Gradient Boosting)

**Anomaly Detection** — Finds unusual patterns without labels
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)

**Ensemble Methods** — Combines multiple models for better accuracy
- Hard Voting
- Soft Voting
- Stacking (meta-learner)
- Hybrid (supervised + anomaly scores)

**Deep Learning** — Neural network-based detection
- Autoencoder (TensorFlow/Keras) with reconstruction error scoring

### Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique) for balanced training
- Configurable contamination rates for anomaly detection models

### Intelligence Dashboard
- **Radar Chart Model Fingerprints** — Visual comparison of model strengths across metrics
- **Performance Grade Ring** — At-a-glance model quality score
- **Leaderboard** — Ranked model comparison with medal indicators
- **Confusion Matrix Heatmaps** — Interactive per-model error analysis
- **ROC Curves** — AUC comparison across all trained models
- **Feature Influence Map** — Which features drive predictions most
- **Anomaly Scatter Plots** — Visual distribution of detected anomalies

### Quick Check (Live Testing)
- **One-Click Quick Start** — Auto-loads demo data, trains models, and opens the testing form
- **Context-Aware Smart Form** — Dollar inputs for amounts, time pickers for hours, day/month selectors, risk sliders, frequency counters, Yes/No toggles
- **Scenario Generators** — Pre-built test cases: Typical/Normal, Looks Suspicious, Very Unusual, Mixed Signals
- **Plain English Summary** — Shows what you're checking in readable format ($150.00, 3:00 PM, Tuesday, Risk: High)
- **Gauge Risk Meter** — Visual risk score from all models combined
- **Model-by-Model Verdicts** — See how each AI model voted independently

### Reports & Model Persistence
- Download styled **HTML reports** with performance metrics and rankings
- Export **CSV reports** with all model metrics
- **Save trained models** to disk (joblib + Keras)
- **Load saved models** for inference without retraining

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Language** | Python | 3.11 |
| **Framework** | Streamlit | 1.54.0 |
| **ML** | scikit-learn | 1.8.0 |
| **Boosting** | XGBoost | 2.1.4 |
| **Deep Learning** | TensorFlow / Keras | 2.20.0 |
| **Imbalance** | imbalanced-learn (SMOTE) | 0.14.1 |
| **Visualization** | Plotly | 5.24.1 |
| **Data** | Pandas / NumPy / SciPy | 2.3.3 / 1.26.4 / 1.17.0 |

---

## Project Structure

```
.
├── app.py                    # Main application (~2,900 lines)
├── requirements.txt          # Pinned Python dependencies
├── pyproject.toml            # Build system configuration
├── replit.md                 # Internal project documentation
├── .gitignore                # Git ignore rules
├── .streamlit/
│   └── config.toml           # Streamlit theme & server config
└── saved_models/             # Persisted trained models (gitignored)
```

---

## Getting Started

### Prerequisites
- Python 3.11+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-anomaly-detector.git
cd ai-anomaly-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py --server.port 5000
```

The app will open at `http://localhost:5000`.

### Quick Start (No Setup Needed)
1. Open the app
2. Click **Quick Check** on the home page
3. Click **Set Up AI & Start Checking** — the AI loads sample data and trains automatically
4. Fill in the smart form or use a scenario generator
5. See your risk score and model verdicts

### Full Analysis Workflow
1. Click **Full Analysis** on the home page
2. Upload a CSV/Excel file or load the demo dataset
3. Review auto-detected column types and data overview
4. Go to **AI Analysis** and train the models you want
5. Explore results in the **Dashboard** (radar charts, ROC curves, leaderboard)
6. Test individual records in **Quick Check**
7. Download reports and save models in **Reports & Models**

---

## How It Works

### Data Pipeline

```
Raw Data (CSV/Excel)
    │
    ▼
Schema Inference ──── Auto-detect column types
    │                 (numeric, categorical, datetime, binary, ID)
    ▼
Preprocessing ─────── Handle missing values, encode categoricals,
    │                 extract datetime features, scale numerics
    ▼
Train/Test Split ──── Stratified split preserving class distribution
    │
    ▼
Model Training ────── Supervised, Anomaly, Ensemble, Deep Learning
    │
    ▼
Evaluation ────────── Accuracy, Precision, Recall, F1, AUC-ROC
    │
    ▼
Live Testing ──────── Score new records against all trained models
```

### Supervised vs Unsupervised Modes

The app automatically adapts based on your data:
- **Supervised mode** (data has labels): Trains classification models that learn what fraud/anomalies look like from examples
- **Unsupervised mode** (no labels): Trains anomaly detection models that find statistical outliers without being told what to look for

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Single-file architecture | `app.py` | Self-contained deployment, easy to share and review |
| Framework | Streamlit | Rapid ML dashboard development with minimal frontend code |
| Chart library | Plotly | Interactive, dark-themed charts with hover details |
| Ensemble strategy | Voting + Stacking + Hybrid | Multiple combination approaches for robust detection |
| UI design | Dark glassmorphism | Modern 2026-style aesthetic suitable for portfolio presentation |
| Two user paths | Quick Check + Full Analysis | Accessible to both non-technical users and data professionals |

---

## Skills Demonstrated

- **Machine Learning**: Supervised classification, unsupervised anomaly detection, ensemble methods, deep learning autoencoders
- **Data Engineering**: Universal preprocessing pipeline, schema inference, feature engineering, handling mixed data types
- **MLOps**: Model persistence, model comparison, performance evaluation, report generation
- **Software Engineering**: ~2,900 lines of clean Python, modular helper functions, robust error handling, edge case guards
- **UI/UX Design**: Dark glassmorphism theme, context-aware smart forms, plain English explanations, responsive layout
- **Data Visualization**: Radar charts, gauge meters, confusion matrix heatmaps, ROC curves, scatter plots, feature importance maps

---
