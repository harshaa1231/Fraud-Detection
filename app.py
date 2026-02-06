import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    page_title="Fraud Detection & Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)


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
        'Time': np.concatenate([normal_time, fraud_time]),
        'Amount': np.concatenate([normal_amount, fraud_amount]),
        'V1': np.concatenate([normal_v1, fraud_v1]),
        'V2': np.concatenate([normal_v2, fraud_v2]),
        'V3': np.concatenate([normal_v3, fraud_v3]),
        'V4': np.concatenate([normal_v4, fraud_v4]),
        'Class': np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    })

    data = data.sample(frac=1).reset_index(drop=True)
    return data


def display_dataset_info():
    st.subheader("Best Fraud Detection Datasets Online")
    datasets_info = [
        {
            "name": "IEEE-CIS Fraud Detection",
            "source": "Kaggle",
            "size": "590K transactions",
            "fraud_rate": "3.5%",
            "link": "https://www.kaggle.com/c/ieee-fraud-detection",
            "description": "Real-world card-not-present transactions with 393 features",
            "best_for": "Production ML models, advanced feature engineering"
        },
        {
            "name": "Credit Card Fraud Detection (European)",
            "source": "Kaggle",
            "size": "285K transactions",
            "fraud_rate": "0.17%",
            "link": "https://www.kaggle.com/mlg-ulb/creditcardfraud",
            "description": "Real anonymized credit card transactions (highly imbalanced)",
            "best_for": "Imbalanced data handling, SMOTE experiments"
        },
        {
            "name": "Credit Card Transactions (Synthetic)",
            "source": "Kaggle",
            "size": "1.2M+ transactions",
            "fraud_rate": "Varies",
            "link": "https://www.kaggle.com/datasets/kartik2112/fraud-detection",
            "description": "Synthetic data with rich features (merchant, demographics)",
            "best_for": "Feature engineering, prototyping"
        },
        {
            "name": "PaySim Financial Dataset",
            "source": "Kaggle",
            "size": "6.3M transactions",
            "fraud_rate": "~0.1%",
            "link": "https://www.kaggle.com/datasets/ealaxi/paysim1",
            "description": "Mobile money transaction simulation",
            "best_for": "Mobile payment fraud detection"
        },
        {
            "name": "Fraud Detection Dataset",
            "source": "Kaggle",
            "size": "Varies",
            "fraud_rate": "Varies",
            "link": "https://www.kaggle.com/datasets/goyaladi/fraud-detection-dataset",
            "description": "General fraud detection scenarios with multiple features",
            "best_for": "Multi-domain fraud detection experiments"
        },
        {
            "name": "German Credit Dataset",
            "source": "UCI ML Repository",
            "size": "1K records",
            "fraud_rate": "30%",
            "link": "https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)",
            "description": "Credit risk assessment with categorical and numerical features",
            "best_for": "Small-scale benchmarking, credit risk"
        }
    ]

    for dataset in datasets_info:
        with st.expander(f"{dataset['name']} ({dataset['source']})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Size:** {dataset['size']}")
                st.write(f"**Fraud Rate:** {dataset['fraud_rate']}")
                st.write(f"**Best For:** {dataset['best_for']}")
            with col2:
                st.write(f"**Description:** {dataset['description']}")
                st.markdown(f"[Download from {dataset['source']}]({dataset['link']})")


def preprocess_data(df, target_column='Class'):
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found. Available columns: {', '.join(df.columns)}")
        return None, None, None, None, None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = X.fillna(X.median())

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    if len(X.columns) == 0:
        st.error("No numeric features found in the dataset after preprocessing.")
        return None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler


def train_supervised_models(X_train, X_test, y_train, y_test, use_smote=False):
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        st.info(f"Applied SMOTE: Training set now has {sum(y_train==0)} normal and {sum(y_train==1)} fraud cases")

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
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

        progress_bar.progress((idx + 1) / len(models))

    status_text.text("All supervised models trained successfully!")
    return results


def train_anomaly_models(X_train, X_test, y_train, y_test, contamination=0.02):
    X_train_normal = X_train[y_train == 0]
    st.info(f"Training anomaly models on {len(X_train_normal):,} normal transactions (excluding fraud cases)")

    models = {
        'Isolation Forest': IsolationForest(contamination=contamination, random_state=42, n_jobs=-1),
        'One-Class SVM': OneClassSVM(nu=contamination, kernel='rbf', gamma='auto'),
    }

    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        model.fit(X_train_normal)

        y_pred = model.predict(X_test)
        y_pred_binary = np.where(y_pred == -1, 1, 0)

        if hasattr(model, 'decision_function'):
            anomaly_scores = -model.decision_function(X_test)
        else:
            anomaly_scores = -model.score_samples(X_test)

        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1': f1_score(y_test, y_pred_binary, zero_division=0),
            'roc_auc': roc_auc_score(y_test, anomaly_scores),
            'predictions': y_pred_binary,
            'anomaly_scores': anomaly_scores,
            'confusion_matrix': confusion_matrix(y_test, y_pred_binary)
        }

        progress_bar.progress((idx + 1) / len(models))

    status_text.text("Training Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination=contamination, novelty=True, n_jobs=-1)
    lof.fit(X_train_normal)

    y_pred_lof = lof.predict(X_test)
    y_pred_lof_binary = np.where(y_pred_lof == -1, 1, 0)
    anomaly_scores_lof = -lof.decision_function(X_test)

    results['Local Outlier Factor'] = {
        'model': lof,
        'accuracy': accuracy_score(y_test, y_pred_lof_binary),
        'precision': precision_score(y_test, y_pred_lof_binary, zero_division=0),
        'recall': recall_score(y_test, y_pred_lof_binary, zero_division=0),
        'f1': f1_score(y_test, y_pred_lof_binary, zero_division=0),
        'roc_auc': roc_auc_score(y_test, anomaly_scores_lof),
        'predictions': y_pred_lof_binary,
        'anomaly_scores': anomaly_scores_lof,
        'confusion_matrix': confusion_matrix(y_test, y_pred_lof_binary)
    }

    progress_bar.progress(1.0)
    status_text.text("All anomaly detection models trained successfully!")
    return results


def train_ensemble_model(X_train, X_test, y_train, y_test, use_smote=False):
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    base_estimators = [
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ('xgb', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'))
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Training Voting Ensemble (Hard Voting)...")
    voting_hard = VotingClassifier(estimators=base_estimators, voting='hard', n_jobs=-1)
    voting_hard.fit(X_train, y_train)
    y_pred_hard = voting_hard.predict(X_test)
    progress_bar.progress(0.25)

    status_text.text("Training Voting Ensemble (Soft Voting)...")
    voting_soft = VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=-1)
    voting_soft.fit(X_train, y_train)
    y_pred_soft = voting_soft.predict(X_test)
    y_proba_soft = voting_soft.predict_proba(X_test)[:, 1]
    progress_bar.progress(0.5)

    status_text.text("Training Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    y_pred_stack = stacking.predict(X_test)
    y_proba_stack = stacking.predict_proba(X_test)[:, 1]
    progress_bar.progress(0.75)

    status_text.text("Training Hybrid Ensemble (Supervised + Anomaly Scores)...")
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
    progress_bar.progress(1.0)

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

    status_text.text("All ensemble models trained successfully!")
    return results


def train_autoencoder(X_train, X_test, y_train, y_test, encoding_dim=None, epochs=50, threshold_percentile=95):
    import tensorflow as tf
    from tensorflow import keras

    tf.get_logger().setLevel('ERROR')

    X_train_normal = X_train[y_train == 0]
    n_features = X_train.shape[1]
    if encoding_dim is None:
        encoding_dim = max(2, n_features // 3)

    st.info(f"Training autoencoder on {len(X_train_normal):,} normal transactions | Input: {n_features} features | Bottleneck: {encoding_dim} neurons")

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
            status_text.text(f"Epoch {epoch + 1}/{epochs} | Loss: {logs.get('loss', 0):.6f} | Val Loss: {logs.get('val_loss', 0):.6f}")

    history = autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=epochs,
        batch_size=32,
        validation_split=0.15,
        verbose=0,
        callbacks=[
            StreamlitCallback(),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    train_reconstructions = autoencoder.predict(X_train_normal, verbose=0)
    train_mse = np.mean(np.power(X_train_normal - train_reconstructions, 2), axis=1)
    threshold = np.percentile(train_mse, threshold_percentile)

    test_reconstructions = autoencoder.predict(X_test, verbose=0)
    test_mse = np.mean(np.power(X_test - test_reconstructions, 2), axis=1)
    y_pred = (test_mse > threshold).astype(int)

    results = {
        'Autoencoder': {
            'model': autoencoder,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, test_mse),
            'predictions': y_pred,
            'anomaly_scores': test_mse,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'threshold': threshold,
            'history': history.history,
            'train_mse': train_mse
        }
    }

    status_text.text("Autoencoder trained successfully!")
    return results


def save_models(model_name, results, scaler, feature_names, preprocessed_data=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    artifacts = {}

    for name, res in results.items():
        model = res['model']
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")

        if hasattr(model, 'save'):
            model.save(os.path.join(save_dir, f"{safe_name}_model.keras"))
        else:
            joblib.dump(model, os.path.join(save_dir, f"{safe_name}_model.joblib"))

        model_artifacts = {}

        if 'iso_forest' in res:
            joblib.dump(res['iso_forest'], os.path.join(save_dir, f"{safe_name}_iso_forest.joblib"))
            model_artifacts['has_iso_forest'] = True

        if 'threshold' in res:
            model_artifacts['threshold'] = float(res['threshold'])

        if model_artifacts:
            artifacts[name] = model_artifacts

    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(list(feature_names), os.path.join(save_dir, "feature_names.joblib"))

    metadata = {
        'timestamp': timestamp,
        'model_type': model_name,
        'models': list(results.keys()),
        'n_features': len(feature_names),
        'feature_names': list(feature_names),
        'metrics': {},
        'artifacts': artifacts
    }
    for name, res in results.items():
        metadata['metrics'][name] = {
            'accuracy': float(res['accuracy']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'f1': float(res['f1']),
            'roc_auc': float(res['roc_auc'])
        }

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


def generate_report_html(supervised_results, anomaly_results, ensemble_results, autoencoder_results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Fraud Detection Report - {timestamp}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
.header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
.header h1 {{ margin: 0; font-size: 28px; }}
.header p {{ margin: 5px 0 0 0; opacity: 0.8; }}
.section {{ background: white; padding: 25px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
.section h2 {{ color: #1a1a2e; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }}
table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
th {{ background: #1a1a2e; color: white; padding: 12px 15px; text-align: left; }}
td {{ padding: 10px 15px; border-bottom: 1px solid #e9ecef; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.metric-good {{ color: #28a745; font-weight: bold; }}
.metric-ok {{ color: #ffc107; font-weight: bold; }}
.metric-bad {{ color: #dc3545; font-weight: bold; }}
.summary-box {{ display: inline-block; background: #e3f2fd; padding: 15px 25px; border-radius: 8px; margin: 5px; text-align: center; }}
.summary-box .value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
.summary-box .label {{ font-size: 12px; color: #666; }}
</style>
</head>
<body>
<div class="header">
<h1>Fraud Detection & Anomaly Detection Report</h1>
<p>Generated: {timestamp}</p>
</div>
"""

    def add_metrics_table(results, section_title):
        if not results:
            return ""
        section = f'<div class="section"><h2>{section_title}</h2><table>'
        section += '<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr>'
        for name, res in results.items():
            def color_class(val):
                if val >= 0.9:
                    return 'metric-good'
                elif val >= 0.7:
                    return 'metric-ok'
                return 'metric-bad'

            section += f'<tr><td><strong>{name}</strong></td>'
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                val = res[metric]
                section += f'<td class="{color_class(val)}">{val:.4f}</td>'
            section += '</tr>'
        section += '</table></div>'
        return section

    all_results = {}
    if supervised_results:
        all_results.update(supervised_results)
    if anomaly_results:
        all_results.update(anomaly_results)
    if ensemble_results:
        all_results.update(ensemble_results)
    if autoencoder_results:
        all_results.update(autoencoder_results)

    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1]['f1'])
        best_auc = max(all_results.items(), key=lambda x: x[1]['roc_auc'])

        html += '<div class="section"><h2>Summary</h2>'
        html += f'<div class="summary-box"><div class="value">{len(all_results)}</div><div class="label">Models Trained</div></div>'
        html += f'<div class="summary-box"><div class="value">{best_model[0]}</div><div class="label">Best F1 Model</div></div>'
        html += f'<div class="summary-box"><div class="value">{best_model[1]["f1"]:.4f}</div><div class="label">Best F1 Score</div></div>'
        html += f'<div class="summary-box"><div class="value">{best_auc[1]["roc_auc"]:.4f}</div><div class="label">Best ROC-AUC</div></div>'
        html += '</div>'

    html += add_metrics_table(supervised_results, "Supervised Models")
    html += add_metrics_table(anomaly_results, "Anomaly Detection Models")
    html += add_metrics_table(ensemble_results, "Ensemble Models")
    html += add_metrics_table(autoencoder_results, "Autoencoder Model")

    if all_results:
        html += '<div class="section"><h2>Model Rankings</h2>'
        ranked = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        html += '<table><tr><th>Rank</th><th>Model</th><th>F1-Score</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th></tr>'
        for rank, (name, res) in enumerate(ranked, 1):
            html += f'<tr><td>{rank}</td><td><strong>{name}</strong></td>'
            html += f'<td>{res["f1"]:.4f}</td><td>{res["roc_auc"]:.4f}</td>'
            html += f'<td>{res["precision"]:.4f}</td><td>{res["recall"]:.4f}</td></tr>'
        html += '</table></div>'

    html += '</body></html>'
    return html


def generate_report_csv(supervised_results, anomaly_results, ensemble_results, autoencoder_results):
    all_results = {}
    if supervised_results:
        for name, res in supervised_results.items():
            all_results[name] = {**{k: v for k, v in res.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}, 'type': 'Supervised'}
    if anomaly_results:
        for name, res in anomaly_results.items():
            all_results[name] = {**{k: v for k, v in res.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}, 'type': 'Anomaly Detection'}
    if ensemble_results:
        for name, res in ensemble_results.items():
            all_results[name] = {**{k: v for k, v in res.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}, 'type': 'Ensemble'}
    if autoencoder_results:
        for name, res in autoencoder_results.items():
            all_results[name] = {**{k: v for k, v in res.items() if k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}, 'type': 'Autoencoder'}

    df = pd.DataFrame(all_results).T
    df.index.name = 'Model'
    return df.to_csv()


def plot_model_comparison(supervised_results, anomaly_results, ensemble_results=None, autoencoder_results=None):
    all_results = {}
    if supervised_results:
        all_results.update(supervised_results)
    if anomaly_results:
        all_results.update(anomaly_results)
    if ensemble_results:
        all_results.update(ensemble_results)
    if autoencoder_results:
        all_results.update(autoencoder_results)

    metrics_df = pd.DataFrame({
        name: {
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1'],
            'ROC-AUC': res['roc_auc']
        }
        for name, res in all_results.items()
    }).T

    fig = go.Figure()
    for metric in metrics_df.columns:
        fig.add_trace(go.Bar(
            name=metric,
            x=metrics_df.index,
            y=metrics_df[metric],
            text=metrics_df[metric].round(3),
            textposition='auto',
        ))

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Detailed Metrics")
    styled_df = metrics_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.4f}")
    st.dataframe(styled_df, width='stretch')

    return metrics_df


def plot_confusion_matrices(results, model_type):
    n_models = len(results)
    cols = st.columns(min(n_models, 3))

    for idx, (name, res) in enumerate(results.items()):
        with cols[idx % 3]:
            cm = res['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Normal', 'Fraud'],
                y=['Normal', 'Fraud'],
                text_auto=True,
                color_continuous_scale='Blues',
                title=f'{name}'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width='stretch')


def plot_feature_importance(supervised_results, feature_names):
    st.subheader("Feature Importance")
    for name, res in supervised_results.items():
        if hasattr(res['model'], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': res['model'].feature_importances_
            }).sort_values('Importance', ascending=False).head(10)

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f'{name} - Top 10 Features',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')


def plot_roc_curves(results, y_test):
    fig = go.Figure()

    for name, res in results.items():
        if 'probabilities' in res:
            fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
        elif 'anomaly_scores' in res:
            fpr, tpr, _ = roc_curve(y_test, res['anomaly_scores'])
        else:
            continue
        auc = res['roc_auc']
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {auc:.3f})',
            mode='lines'
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title='ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')


def plot_anomaly_scores(anomaly_results, y_test):
    for name, res in anomaly_results.items():
        scores = res['anomaly_scores']
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores[y_test == 0], name='Normal', opacity=0.7, nbinsx=50))
        fig.add_trace(go.Histogram(x=scores[y_test == 1], name='Fraud', opacity=0.7, nbinsx=50))
        fig.update_layout(
            title=f'{name} - Anomaly Score Distribution',
            xaxis_title='Anomaly Score',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, width='stretch')


def main():
    st.title("Fraud Detection & Anomaly Detection System")
    st.markdown("**Advanced Machine Learning Platform for Fraud Detection with Supervised, Unsupervised, Ensemble, and Deep Learning Models**")

    st.sidebar.header("Configuration")

    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Sample Dataset", "Upload Your Own Dataset", "View Available Datasets"]
    )

    if data_source == "View Available Datasets":
        display_dataset_info()
        st.info("Download any of the datasets above and upload them using the 'Upload Your Own Dataset' option.")
        return

    if data_source == "Use Sample Dataset":
        with st.spinner("Generating sample dataset..."):
            df = generate_sample_data(n_samples=10000, fraud_ratio=0.02)
        st.success(f"Sample dataset loaded: {len(df)} transactions")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        if uploaded_file is None:
            st.info("Please upload a dataset to begin analysis")
            return
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Dataset loaded: {len(df)} records")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return

    target_column = st.sidebar.selectbox(
        "Select Target Column (Fraud Label):",
        df.columns,
        index=len(df.columns) - 1 if 'Class' not in df.columns else df.columns.tolist().index('Class')
    )

    tabs = st.tabs([
        "Data Exploration",
        "Supervised Models",
        "Anomaly Detection",
        "Ensemble Models",
        "Autoencoder",
        "Model Comparison",
        "Reports & Export",
        "Model Storage",
        "Make Predictions"
    ])

    # ---- Tab 1: Data Exploration ----
    with tabs[0]:
        st.header("Data Exploration")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            fraud_count = df[target_column].sum()
            st.metric("Fraud Cases", f"{int(fraud_count):,}")
        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            st.metric("Features", len(df.columns) - 1)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), width='stretch')

        st.subheader("Class Distribution")
        class_counts = df[target_column].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=['Normal', 'Fraud'],
                y=[class_counts.get(0, 0), class_counts.get(1, 0)],
                text=[class_counts.get(0, 0), class_counts.get(1, 0)],
                textposition='auto',
                marker_color=['#2ecc71', '#e74c3c']
            )
        ])
        fig.update_layout(title='Transaction Distribution', xaxis_title='Class', yaxis_title='Count', height=400)
        st.plotly_chart(fig, width='stretch')

        st.subheader("Feature Statistics")
        st.dataframe(df.describe(), width='stretch')

        st.subheader("Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) <= 15:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title='Feature Correlation Matrix')
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info(f"Correlation matrix skipped for {len(numeric_df.columns)} features. Showing top correlations with target instead.")
            if target_column in numeric_df.columns:
                target_corr = numeric_df.corr()[target_column].drop(target_column).abs().sort_values(ascending=False).head(15)
                fig = px.bar(x=target_corr.index, y=target_corr.values, labels={'x': 'Feature', 'y': 'Correlation with Target'}, title='Top 15 Feature Correlations with Target')
                st.plotly_chart(fig, width='stretch')

    # ---- Tab 2: Supervised Models ----
    with tabs[1]:
        st.header("Supervised Fraud Detection Models")
        st.markdown("Train classification models using labeled fraud data")

        use_smote = st.checkbox(
            "Apply SMOTE (Synthetic Minority Over-sampling)",
            value=True,
            help="Balances the dataset by creating synthetic fraud samples"
        )

        if st.button("Train Supervised Models", type="primary", key="train_supervised"):
            with st.spinner("Preprocessing data..."):
                result = preprocess_data(df, target_column)
                if result[0] is None:
                    st.stop()
                X_train, X_test, y_train, y_test, feature_names, scaler = result
                st.session_state['preprocessed_data'] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test,
                    'feature_names': feature_names, 'scaler': scaler
                }

            with st.spinner("Training models..."):
                supervised_results = train_supervised_models(X_train, X_test, y_train, y_test, use_smote)
                st.session_state['supervised_results'] = supervised_results
            st.success("Models trained successfully!")

        if 'supervised_results' in st.session_state:
            results = st.session_state['supervised_results']
            y_test = st.session_state['preprocessed_data']['y_test']
            feature_names = st.session_state['preprocessed_data']['feature_names']

            st.subheader("Model Performance")
            metrics_df = pd.DataFrame({
                name: {'Accuracy': res['accuracy'], 'Precision': res['precision'],
                       'Recall': res['recall'], 'F1-Score': res['f1'], 'ROC-AUC': res['roc_auc']}
                for name, res in results.items()
            }).T
            styled_df = metrics_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.4f}")
            st.dataframe(styled_df, width='stretch')

            st.subheader("ROC Curves")
            plot_roc_curves(results, y_test)

            st.subheader("Confusion Matrices")
            plot_confusion_matrices(results, "Supervised")

            plot_feature_importance(results, feature_names)

    # ---- Tab 3: Anomaly Detection ----
    with tabs[2]:
        st.header("Unsupervised Anomaly Detection Models")
        st.markdown("Detect fraud using unsupervised learning (no labels required)")

        contamination = st.slider(
            "Contamination Rate (Expected Fraud %)",
            min_value=0.01, max_value=0.20, value=0.02, step=0.01,
            help="Expected proportion of anomalies in the dataset"
        )

        if st.button("Train Anomaly Detection Models", type="primary", key="train_anomaly"):
            if 'preprocessed_data' not in st.session_state:
                with st.spinner("Preprocessing data..."):
                    result = preprocess_data(df, target_column)
                    if result[0] is None:
                        st.stop()
                    X_train, X_test, y_train, y_test, feature_names, scaler = result
                    st.session_state['preprocessed_data'] = {
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test,
                        'feature_names': feature_names, 'scaler': scaler
                    }

            ppd = st.session_state['preprocessed_data']
            with st.spinner("Training anomaly detection models..."):
                anomaly_results = train_anomaly_models(
                    ppd['X_train'], ppd['X_test'], ppd['y_train'], ppd['y_test'], contamination
                )
                st.session_state['anomaly_results'] = anomaly_results
            st.success("Anomaly detection models trained successfully!")

        if 'anomaly_results' in st.session_state:
            results = st.session_state['anomaly_results']
            y_test = st.session_state['preprocessed_data']['y_test']

            st.subheader("Model Performance")
            metrics_df = pd.DataFrame({
                name: {'Accuracy': res['accuracy'], 'Precision': res['precision'],
                       'Recall': res['recall'], 'F1-Score': res['f1'], 'ROC-AUC': res['roc_auc']}
                for name, res in results.items()
            }).T
            styled_df = metrics_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.4f}")
            st.dataframe(styled_df, width='stretch')

            st.subheader("Confusion Matrices")
            plot_confusion_matrices(results, "Anomaly Detection")

            st.subheader("Anomaly Score Distributions")
            plot_anomaly_scores(results, y_test)

    # ---- Tab 4: Ensemble Models ----
    with tabs[3]:
        st.header("Ensemble Model Stacking")
        st.markdown("Combine multiple models for improved prediction accuracy")

        st.markdown("""
**Ensemble methods included:**
- **Voting (Hard):** Each model votes, majority wins
- **Voting (Soft):** Averages probability predictions across models
- **Stacking:** Trains a meta-learner on base model predictions
- **Hybrid:** Combines supervised models with anomaly detection scores
""")

        use_smote_ensemble = st.checkbox(
            "Apply SMOTE for Ensemble Training",
            value=True,
            key="smote_ensemble"
        )

        if st.button("Train Ensemble Models", type="primary", key="train_ensemble"):
            if 'preprocessed_data' not in st.session_state:
                with st.spinner("Preprocessing data..."):
                    result = preprocess_data(df, target_column)
                    if result[0] is None:
                        st.stop()
                    X_train, X_test, y_train, y_test, feature_names, scaler = result
                    st.session_state['preprocessed_data'] = {
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test,
                        'feature_names': feature_names, 'scaler': scaler
                    }

            ppd = st.session_state['preprocessed_data']
            with st.spinner("Training ensemble models..."):
                ensemble_results = train_ensemble_model(
                    ppd['X_train'], ppd['X_test'], ppd['y_train'], ppd['y_test'],
                    use_smote_ensemble
                )
                st.session_state['ensemble_results'] = ensemble_results
            st.success("Ensemble models trained successfully!")

        if 'ensemble_results' in st.session_state:
            results = st.session_state['ensemble_results']
            y_test = st.session_state['preprocessed_data']['y_test']

            st.subheader("Ensemble Model Performance")
            metrics_df = pd.DataFrame({
                name: {'Accuracy': res['accuracy'], 'Precision': res['precision'],
                       'Recall': res['recall'], 'F1-Score': res['f1'], 'ROC-AUC': res['roc_auc']}
                for name, res in results.items()
            }).T
            styled_df = metrics_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.4f}")
            st.dataframe(styled_df, width='stretch')

            st.subheader("Confusion Matrices")
            plot_confusion_matrices(results, "Ensemble")

            roc_results = {k: v for k, v in results.items() if 'probabilities' in v}
            if roc_results:
                st.subheader("ROC Curves")
                plot_roc_curves(roc_results, y_test)

    # ---- Tab 5: Autoencoder ----
    with tabs[4]:
        st.header("Autoencoder Anomaly Detection")
        st.markdown("Deep learning-based anomaly detection using neural network autoencoders")

        st.markdown("""
**How it works:** An autoencoder learns to reconstruct normal transactions.
Fraudulent transactions produce higher reconstruction errors, making them detectable as anomalies.
""")

        col1, col2, col3 = st.columns(3)
        with col1:
            ae_epochs = st.slider("Training Epochs", 10, 200, 50, step=10, key="ae_epochs")
        with col2:
            ae_threshold = st.slider("Anomaly Threshold Percentile", 85, 99, 95, key="ae_threshold")
        with col3:
            ae_encoding = st.slider("Encoding Dimension", 2, 20, 4, key="ae_encoding")

        if st.button("Train Autoencoder", type="primary", key="train_autoencoder"):
            if 'preprocessed_data' not in st.session_state:
                with st.spinner("Preprocessing data..."):
                    result = preprocess_data(df, target_column)
                    if result[0] is None:
                        st.stop()
                    X_train, X_test, y_train, y_test, feature_names, scaler = result
                    st.session_state['preprocessed_data'] = {
                        'X_train': X_train, 'X_test': X_test,
                        'y_train': y_train, 'y_test': y_test,
                        'feature_names': feature_names, 'scaler': scaler
                    }

            ppd = st.session_state['preprocessed_data']
            with st.spinner("Training autoencoder (this may take a moment)..."):
                autoencoder_results = train_autoencoder(
                    ppd['X_train'], ppd['X_test'], ppd['y_train'], ppd['y_test'],
                    encoding_dim=ae_encoding, epochs=ae_epochs, threshold_percentile=ae_threshold
                )
                st.session_state['autoencoder_results'] = autoencoder_results
            st.success("Autoencoder trained successfully!")

        if 'autoencoder_results' in st.session_state:
            results = st.session_state['autoencoder_results']
            y_test = st.session_state['preprocessed_data']['y_test']
            ae_res = results['Autoencoder']

            st.subheader("Autoencoder Performance")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{ae_res['accuracy']:.4f}")
            col2.metric("Precision", f"{ae_res['precision']:.4f}")
            col3.metric("Recall", f"{ae_res['recall']:.4f}")
            col4.metric("F1-Score", f"{ae_res['f1']:.4f}")
            col5.metric("ROC-AUC", f"{ae_res['roc_auc']:.4f}")

            st.subheader("Confusion Matrix")
            cm = ae_res['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Normal', 'Fraud'], y=['Normal', 'Fraud'],
                text_auto=True, color_continuous_scale='Blues',
                title='Autoencoder Confusion Matrix'
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')

            if 'history' in ae_res:
                st.subheader("Training History")
                history = ae_res['history']
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history['loss'], name='Training Loss', mode='lines'))
                if 'val_loss' in history:
                    fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss', mode='lines'))
                fig.update_layout(title='Autoencoder Training Loss', xaxis_title='Epoch', yaxis_title='MSE Loss', height=400)
                st.plotly_chart(fig, width='stretch')

            st.subheader("Reconstruction Error Distribution")
            scores = ae_res['anomaly_scores']
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=scores[y_test == 0], name='Normal', opacity=0.7, nbinsx=50))
            fig.add_trace(go.Histogram(x=scores[y_test == 1], name='Fraud', opacity=0.7, nbinsx=50))
            fig.add_vline(x=ae_res['threshold'], line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(
                title='Reconstruction Error Distribution',
                xaxis_title='Reconstruction Error (MSE)',
                yaxis_title='Count',
                barmode='overlay', height=400
            )
            st.plotly_chart(fig, width='stretch')

    # ---- Tab 6: Model Comparison ----
    with tabs[5]:
        st.header("Comprehensive Model Comparison")

        has_any = any(k in st.session_state for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'])
        if has_any:
            sup = st.session_state.get('supervised_results', {})
            anom = st.session_state.get('anomaly_results', {})
            ens = st.session_state.get('ensemble_results', {})
            ae = st.session_state.get('autoencoder_results', {})

            plot_model_comparison(sup, anom, ens, ae)

            all_results = {**sup, **anom, **ens, **ae}
            if all_results:
                y_test = st.session_state['preprocessed_data']['y_test']
                st.subheader("ROC Curves (All Models)")
                plot_roc_curves(all_results, y_test)

                st.subheader("Best Model Recommendation")
                best_f1 = max(all_results.items(), key=lambda x: x[1]['f1'])
                best_auc = max(all_results.items(), key=lambda x: x[1]['roc_auc'])
                best_recall = max(all_results.items(), key=lambda x: x[1]['recall'])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"**Best F1-Score:** {best_f1[0]} ({best_f1[1]['f1']:.4f})")
                with col2:
                    st.success(f"**Best ROC-AUC:** {best_auc[0]} ({best_auc[1]['roc_auc']:.4f})")
                with col3:
                    st.success(f"**Best Recall:** {best_recall[0]} ({best_recall[1]['recall']:.4f})")
        else:
            st.info("Train models in the previous tabs to see comparison")

    # ---- Tab 7: Reports & Export ----
    with tabs[6]:
        st.header("Reports & Export")

        has_any = any(k in st.session_state for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'])
        if not has_any:
            st.info("Train models first to generate reports")
        else:
            sup = st.session_state.get('supervised_results', {})
            anom = st.session_state.get('anomaly_results', {})
            ens = st.session_state.get('ensemble_results', {})
            ae = st.session_state.get('autoencoder_results', {})

            st.subheader("Download Reports")

            col1, col2 = st.columns(2)
            with col1:
                html_report = generate_report_html(sup, anom, ens, ae)
                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name=f"fraud_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    type="primary"
                )
            with col2:
                csv_report = generate_report_csv(sup, anom, ens, ae)
                st.download_button(
                    label="Download CSV Report",
                    data=csv_report,
                    file_name=f"fraud_detection_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )

            st.subheader("Report Preview")
            st.components.v1.html(html_report, height=600, scrolling=True)

    # ---- Tab 8: Model Storage ----
    with tabs[7]:
        st.header("Model Storage")

        st.subheader("Save Trained Models")
        has_any = any(k in st.session_state for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'])

        if has_any:
            model_types = []
            if 'supervised_results' in st.session_state:
                model_types.append("Supervised")
            if 'anomaly_results' in st.session_state:
                model_types.append("Anomaly Detection")
            if 'ensemble_results' in st.session_state:
                model_types.append("Ensemble")
            if 'autoencoder_results' in st.session_state:
                model_types.append("Autoencoder")

            save_type = st.selectbox("Select model type to save:", model_types)

            if st.button("Save Models", type="primary", key="save_models"):
                result_key = {
                    "Supervised": "supervised_results",
                    "Anomaly Detection": "anomaly_results",
                    "Ensemble": "ensemble_results",
                    "Autoencoder": "autoencoder_results"
                }[save_type]

                results = st.session_state[result_key]
                scaler = st.session_state['preprocessed_data']['scaler']
                feature_names = st.session_state['preprocessed_data']['feature_names']

                save_dir = save_models(save_type, results, scaler, feature_names)
                st.success(f"Models saved to: {save_dir}")
        else:
            st.info("Train models first before saving")

        st.subheader("Load Saved Models")
        saved_models = load_saved_models()

        if saved_models:
            for idx, meta in enumerate(saved_models):
                with st.expander(f"{meta['model_type']} - {meta['timestamp']} ({len(meta['models'])} models)"):
                    st.write(f"**Models:** {', '.join(meta['models'])}")
                    st.write(f"**Features:** {meta['n_features']}")

                    if meta['metrics']:
                        metrics_df = pd.DataFrame(meta['metrics']).T
                        st.dataframe(metrics_df.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.4f}"), width='stretch')

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Load Models", key=f"load_{idx}"):
                            try:
                                models, scaler, feature_names, extras = load_model_from_dir(meta['dir_path'], meta)
                                st.session_state['loaded_models'] = {
                                    'models': models,
                                    'scaler': scaler,
                                    'feature_names': feature_names,
                                    'metadata': meta,
                                    'extras': extras
                                }
                                st.success(f"Loaded {len(models)} models successfully!")
                            except Exception as e:
                                st.error(f"Error loading models: {str(e)}")
                    with col2:
                        if st.button(f"Delete", key=f"delete_{idx}"):
                            import shutil
                            shutil.rmtree(meta['dir_path'])
                            st.success("Models deleted successfully!")
                            st.rerun()
        else:
            st.info("No saved models found")

    # ---- Tab 9: Make Predictions ----
    with tabs[8]:
        st.header("Real-Time Fraud Prediction")

        has_trained = any(k in st.session_state for k in ['supervised_results', 'anomaly_results', 'ensemble_results', 'autoencoder_results'])
        has_loaded = 'loaded_models' in st.session_state

        if not has_trained and not has_loaded:
            st.info("Train or load models first to make predictions")
            st.stop()

        if has_trained:
            feature_names = st.session_state['preprocessed_data']['feature_names']
            scaler = st.session_state['preprocessed_data']['scaler']
        elif has_loaded:
            feature_names = st.session_state['loaded_models']['feature_names']
            scaler = st.session_state['loaded_models']['scaler']
        else:
            return

        st.subheader("Enter Transaction Details")

        input_data = {}
        cols = st.columns(3)
        for idx, feature in enumerate(feature_names):
            with cols[idx % 3]:
                input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f", key=f"pred_{feature}")

        if st.button("Predict Fraud", type="primary", key="predict"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            st.subheader("Prediction Results")

            if 'supervised_results' in st.session_state:
                st.markdown("### Supervised Models")
                for name, res in st.session_state['supervised_results'].items():
                    model = res['model']
                    prediction = model.predict(input_scaled)[0]
                    proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else [0, 0]
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{name}**")
                    with col2:
                        if prediction == 1:
                            st.error("FRAUD DETECTED")
                        else:
                            st.success("NORMAL")
                    with col3:
                        st.write(f"Confidence: {max(proba):.2%}")

            if 'anomaly_results' in st.session_state:
                st.markdown("### Anomaly Detection Models")
                for name, res in st.session_state['anomaly_results'].items():
                    model = res['model']
                    prediction = model.predict(input_scaled)[0]
                    prediction_binary = 1 if prediction == -1 else 0
                    if hasattr(model, 'decision_function'):
                        score = -model.decision_function(input_scaled)[0]
                    else:
                        score = -model.score_samples(input_scaled)[0]
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{name}**")
                    with col2:
                        if prediction_binary == 1:
                            st.error("ANOMALY DETECTED")
                        else:
                            st.success("NORMAL")
                    with col3:
                        st.write(f"Score: {score:.4f}")

            if 'ensemble_results' in st.session_state:
                st.markdown("### Ensemble Models")
                for name, res in st.session_state['ensemble_results'].items():
                    model = res['model']
                    if 'Hybrid' in name and 'iso_forest' in res:
                        iso_score = res['iso_forest'].decision_function(input_scaled).reshape(1, -1)
                        input_hybrid = np.hstack([input_scaled, iso_score])
                        prediction = model.predict(input_hybrid)[0]
                        proba = model.predict_proba(input_hybrid)[0]
                    else:
                        prediction = model.predict(input_scaled)[0]
                        proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else [0, 0]
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{name}**")
                    with col2:
                        if prediction == 1:
                            st.error("FRAUD DETECTED")
                        else:
                            st.success("NORMAL")
                    with col3:
                        st.write(f"Confidence: {max(proba):.2%}")

            if 'autoencoder_results' in st.session_state:
                st.markdown("### Autoencoder")
                ae_res = st.session_state['autoencoder_results']['Autoencoder']
                model = ae_res['model']
                reconstruction = model.predict(input_scaled, verbose=0)
                mse = np.mean(np.power(input_scaled - reconstruction, 2))
                threshold = ae_res['threshold']
                is_anomaly = mse > threshold

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write("**Autoencoder**")
                with col2:
                    if is_anomaly:
                        st.error("ANOMALY DETECTED")
                    else:
                        st.success("NORMAL")
                with col3:
                    st.write(f"Error: {mse:.6f} (Threshold: {threshold:.6f})")

            if has_loaded:
                st.markdown("### Loaded Models")
                loaded = st.session_state['loaded_models']
                extras = loaded.get('extras', {})
                for name, model in loaded['models'].items():
                    try:
                        model_extras = extras.get(name, {})

                        if 'threshold' in model_extras and hasattr(model, 'predict'):
                            reconstruction = model.predict(input_scaled, verbose=0)
                            mse = np.mean(np.power(input_scaled - reconstruction, 2))
                            threshold = model_extras['threshold']
                            is_anomaly = mse > threshold
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{name}**")
                            with col2:
                                if is_anomaly:
                                    st.error("ANOMALY DETECTED")
                                else:
                                    st.success("NORMAL")
                            with col3:
                                st.write(f"Error: {mse:.6f} (Threshold: {threshold:.6f})")

                        elif 'iso_forest' in model_extras and hasattr(model, 'predict_proba'):
                            iso_score = model_extras['iso_forest'].decision_function(input_scaled).reshape(1, -1)
                            input_hybrid = np.hstack([input_scaled, iso_score])
                            prediction = model.predict(input_hybrid)[0]
                            proba = model.predict_proba(input_hybrid)[0]
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{name}**")
                            with col2:
                                if prediction == 1:
                                    st.error("FRAUD DETECTED")
                                else:
                                    st.success("NORMAL")
                            with col3:
                                st.write(f"Confidence: {max(proba):.2%}")

                        elif hasattr(model, 'predict_proba'):
                            prediction = model.predict(input_scaled)[0]
                            proba = model.predict_proba(input_scaled)[0]
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{name}**")
                            with col2:
                                if prediction == 1:
                                    st.error("FRAUD DETECTED")
                                else:
                                    st.success("NORMAL")
                            with col3:
                                st.write(f"Confidence: {max(proba):.2%}")

                        elif hasattr(model, 'predict'):
                            prediction = model.predict(input_scaled)
                            if hasattr(prediction, '__len__'):
                                prediction = prediction[0]
                            pred_binary = 1 if prediction == -1 else int(prediction)
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**{name}**")
                            with col2:
                                if pred_binary == 1:
                                    st.error("ANOMALY DETECTED")
                                else:
                                    st.success("NORMAL")
                    except Exception as e:
                        st.warning(f"Could not predict with {name}: {str(e)}")


if __name__ == "__main__":
    main()
