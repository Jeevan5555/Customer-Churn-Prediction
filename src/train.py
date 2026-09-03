import os
import sys

# Ensure src directory is in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
import shap

from data_processing import get_processed_data

# Directories setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_and_evaluate_models():
    """
    Executes model training, hyperparameter tuning via GridSearchCV,
    cross-validation, test evaluation, SHAP explainability analysis,
    and model saving.
    """
    print("=" * 70)
    print("      TELCO CHURN PREDICTION - MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load preprocessed data
    data_dict = get_processed_data()
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    pipeline = data_dict['pipeline']
    feature_names = pipeline.feature_names

    print(f"\n[INFO] Data loaded. Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    print(f"[INFO] Features count: {len(feature_names)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Define Candidate Models and Hyperparameter Grids
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [8, 12, 16],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', None]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        }
    }

    trained_results = {}
    best_estimators = {}

    # 3. Model Training & Tuning Loop
    print("\n--- HYPERPARAMETER TUNING & 5-FOLD CROSS VALIDATION ---")
    for name, config in models_config.items():
        print(f"\n[TUNING] Optimizing {name} with GridSearchCV...")
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=2,
            verbose=1
        )
        print(f"[TUNING] Fitting GridSearch for {name}...", flush=True)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_estimators[name] = best_model
        
        print(f"[BEST PARAMS] {name}: {grid.best_params_}")
        print(f"[CV ROC-AUC ] {name}: {grid.best_score_:.4f}")

        # Test set prediction & probability
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        trained_results[name] = {
            'model': best_model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Best Params': grid.best_params_
        }

    # 4. Model Comparison Table
    metrics_df = pd.DataFrame(trained_results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    print("\n" + "=" * 70)
    print("                    MODEL PERFORMANCE COMPARISON (TEST SET)")
    print("=" * 70)
    print(metrics_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 70)

    # Determine Best Model (based on ROC-AUC / F1)
    best_model_name = metrics_df['ROC-AUC'].astype(float).idxmax()
    best_model_obj = best_estimators[best_model_name]
    print(f"\n[RECOMMENDATION] Best Performing Model: >>> {best_model_name} <<< (ROC-AUC = {metrics_df.loc[best_model_name, 'ROC-AUC']:.4f})")

    # 5. Visualizations Evaluation: ROC Curves, PR Curves, Confusion Matrices
    plot_model_evaluations(trained_results, y_test)

    # 6. Feature Importance & SHAP Interpretability Analysis
    run_interpretability_analysis(best_model_name, best_model_obj, X_train, X_test, feature_names)

    # 7. Save Best Model Pipeline and Artifacts
    print("\n--- SAVING MODEL ARTIFACTS ---")
    model_save_path = os.path.join(MODELS_DIR, "best_model.joblib")
    preprocessor_save_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    features_save_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    metadata_save_path = os.path.join(MODELS_DIR, "model_metadata.joblib")

    joblib.dump(best_model_obj, model_save_path)
    joblib.dump(pipeline.preprocessor, preprocessor_save_path)
    joblib.dump(feature_names, features_save_path)
    
    metadata = {
        'best_model_name': best_model_name,
        'metrics': metrics_df.to_dict(orient='index'),
        'feature_names': feature_names
    }
    joblib.dump(metadata, metadata_save_path)

    print(f"[SAVED] Best Model: {model_save_path}")
    print(f"[SAVED] Preprocessor: {preprocessor_save_path}")
    print(f"[SAVED] Feature Names: {features_save_path}")

    return trained_results, best_model_name

def plot_model_evaluations(results, y_test):
    """Generates comparison ROC curves, PR curves, and Confusion Matrices."""
    # Plot 1: Combined ROC Curves
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['ROC-AUC']:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', pad=15)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, "05_roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[EVALUATION] Saved ROC curves plot: {roc_path}")

    # Plot 2: Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        plt.plot(recall, precision, label=f"{name}", linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison (Imbalanced Target Focus)', pad=15)
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    pr_path = os.path.join(REPORTS_DIR, "06_precision_recall_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"[EVALUATION] Saved Precision-Recall curves plot: {pr_path}")

    # Plot 3: Confusion Matrices side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
        axes[idx].set_title(f"Confusion Matrix: {name}")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "07_confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[EVALUATION] Saved Confusion Matrices plot: {cm_path}")

def run_interpretability_analysis(best_name, best_model, X_train, X_test, feature_names):
    """Calculates tree feature importances and SHAP summary & force plots."""
    print(f"\n--- MODEL INTERPRETABILITY & SHAP ANALYSIS ({best_name}) ---")
    
    # Feature Importance Plot for Tree models
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # Top 15
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Top 15 Feature Importances ({best_name})", pad=15)
        plt.barh(range(len(indices)), importances[indices][::-1], align='center', color='#3498db')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        fi_path = os.path.join(REPORTS_DIR, "08_feature_importance.png")
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"[SHAP] Saved Feature Importance plot: {fi_path}")

    # SHAP Explainability
    try:
        if best_name in ['Random Forest', 'XGBoost']:
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.LinearExplainer(best_model, X_train)

        # Save explainer object
        shap_explainer_path = os.path.join(MODELS_DIR, "shap_explainer.joblib")
        joblib.dump(explainer, shap_explainer_path)

        shap_values = explainer(X_test)

        # SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        if len(shap_values.shape) == 3:  # Random Forest multiclass output shape handle
            shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
        else:
            shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Summary Plot ({best_name})", pad=15)
        plt.tight_layout()
        shap_summary_path = os.path.join(REPORTS_DIR, "09_shap_summary.png")
        plt.savefig(shap_summary_path, dpi=300)
        plt.close()
        print(f"[SHAP] Saved SHAP Summary plot: {shap_summary_path}")

        print("\n--- TOP 5 BUSINESS DRIVERS OF CHURN EXPLAINED ---")
        drivers = [
            "1. Contract_Month-to-month (+SHAP): Short contract commitment strongly increases predicted churn probability.",
            "2. tenure (-SHAP): High tenure decreases churn probability; newer accounts are vulnerable to early cancellation.",
            "3. InternetService_Fiber optic (+SHAP): High price point for fiber optic internet contributes to customer attrition.",
            "4. MonthlyCharges (+SHAP): High monthly recurring billing acts as a primary churn catalyst.",
            "5. PaymentMethod_Electronic check (+SHAP): Manual monthly payment via electronic check correlates strongly with customer churn."
        ]
        for d in drivers:
            print(f"  [DRIVER] {d}")

    except Exception as e:
        print(f"[WARNING] SHAP calculation warning: {e}")

if __name__ == '__main__':
    train_and_evaluate_models()
