# Fraud Detection & Anomaly Detection Dashboard

## Overview

This project is a **Streamlit-based fraud detection and anomaly detection dashboard**. It generates synthetic transaction data or accepts uploaded datasets, applies multiple machine learning models (supervised, unsupervised, ensemble, and deep learning) to detect fraudulent transactions, and visualizes results through interactive charts. The application includes model persistence, downloadable reports, and real-time prediction capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Structure
- **Single-file Streamlit app** (`app.py`): The entire application lives in one file, combining data generation, model training, evaluation, visualization, model storage, and report generation.
- **`saved_models/` directory**: Stores serialized trained models (joblib/keras format) with metadata for cross-session persistence.

### Key Components

**Data Generation & Upload**
- Synthetic fraud dataset generated with NumPy (no external dataset required)
- CSV/Excel file upload support for custom datasets
- Curated list of best fraud detection datasets with download links (IEEE-CIS, European CC, PaySim, etc.)
- Configurable sample size and fraud ratio (default 2% fraud)

**Machine Learning Models**
- **Supervised models**: Logistic Regression, Random Forest, XGBoost
- **Unsupervised/Anomaly Detection models**: Isolation Forest, One-Class SVM, Local Outlier Factor
- **Ensemble models**: Hard Voting, Soft Voting, Stacking (meta-learner), Hybrid (supervised + anomaly scores)
- **Deep Learning**: Autoencoder-based anomaly detection using TensorFlow/Keras
- **Class imbalance handling**: SMOTE (Synthetic Minority Over-sampling Technique)

**Model Persistence**
- Save trained models to disk using joblib (sklearn models) and Keras save (autoencoders)
- Load previously saved models for inference without retraining
- Metadata tracking (timestamps, metrics, feature names) via JSON

**Reports & Export**
- Downloadable HTML reports with styled performance metrics, rankings, and summary
- Downloadable CSV reports with all model metrics
- In-app report preview

**Visualization**
- Plotly for interactive charts (ROC curves, confusion matrices, feature importance, anomaly scores, training history)
- Feature correlation matrix and target correlation analysis
- Streamlit's wide layout mode for dashboard-style presentation

### How to Run
Run the Streamlit app with:
```bash
streamlit run app.py --server.port 5000
```

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | Streamlit | Rapid prototyping of data science dashboards with minimal frontend code |
| Data source | Synthetic + Upload | No dependency on external datasets; supports custom data too |
| ML library | scikit-learn + XGBoost | Industry-standard for supervised and unsupervised approaches |
| Deep Learning | TensorFlow/Keras | Autoencoder-based anomaly detection with training callbacks |
| Ensemble | sklearn StackingClassifier + VotingClassifier | Built-in support for model combination strategies |
| Model persistence | joblib + Keras save | Efficient serialization for sklearn and deep learning models |
| Imbalance handling | SMOTE via imbalanced-learn | Creates synthetic minority samples to improve model training |
| Charts | Plotly (primary) | Interactive, zoomable charts that work well within Streamlit |
| Reports | HTML + CSV | Professional styled reports for sharing and analysis |

## External Dependencies

### Python Packages
- **streamlit** — Web application framework for the dashboard UI
- **pandas / numpy** — Data manipulation and numerical computation
- **plotly** — Interactive visualizations (primary charting library)
- **matplotlib / seaborn** — Additional static plotting
- **scikit-learn** — ML models, preprocessing, metrics, ensemble methods
- **xgboost** — Gradient boosting classifier
- **imbalanced-learn (imblearn)** — SMOTE oversampling
- **tensorflow / keras** — Autoencoder-based deep learning anomaly detection
- **joblib** — Model serialization and persistence

### External Services
- None — the application is entirely self-contained with no database, API, or external service dependencies

## Recent Changes
- 2026-02-06: Added ensemble model stacking (Voting Hard/Soft, Stacking, Hybrid)
- 2026-02-06: Added autoencoder-based deep learning anomaly detection
- 2026-02-06: Added model persistence (save/load with joblib and Keras)
- 2026-02-06: Added downloadable HTML and CSV reports
- 2026-02-06: Added feature correlation matrix visualization
- 2026-02-06: Added best model recommendation in comparison tab
- 2026-02-06: Expanded dataset info with additional datasets and metadata
