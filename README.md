# Telco Customer Churn Prediction & Risk Scoring Intelligence Prototype

An end-to-end Machine Learning pipeline and interactive Streamlit web dashboard for predicting customer churn, identifying risk factors, and delivering actionable retention insights for telecommunications and subscription-based service providers.

---

## 📌 Problem Statement
Customer retention is a key growth driver for subscription businesses. Acquiring new customers costs 5x to 25x more than retaining existing ones. By accurately predicting customer churn probability and understanding the top operational drivers behind attrition, retention teams can deploy proactive, targeted intervention strategies before customers cancel their service.

---

## 📁 Dataset Description
The system is built on the standard **IBM Telco Customer Churn** dataset (`7,043` records, `21` attributes):
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account & Billing:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Target Variable:** `Churn` (Yes = 1, No = 0)

---

## 🚀 Repository Structure
```
p1/
├── requirements.txt           # Project Python dependencies
├── README.md                  # Comprehensive documentation & setup guide
├── data/
│   ├── download_data.py       # Script to fetch/generate dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_processing.py     # Data cleaning, missing value imputation & feature engineering
│   ├── eda.py                 # Automated EDA & figure generation
│   └── train.py               # Model training, hyperparameter tuning (GridSearchCV), evaluation & SHAP
├── models/                    # Serialized model artifacts (.joblib)
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   ├── metadata.joblib
│   └── shap_explainer.joblib
├── reports/
│   └── figures/               # Generated EDA & evaluation charts (.png)
└── app/
    └── app.py                 # Production Streamlit Web Dashboard
```

---

## 🛠️ Installation & Setup Guide

### 1. Clone & Set Up Environment
```bash
# Navigate to project directory
cd p1

# Install required Python dependencies
pip install -r requirements.txt
```

### 2. Download / Generate Data
```bash
python data/download_data.py
```

### 3. Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
*Generates and saves visual analysis charts to `reports/figures/`.*

### 4. Train, Tune & Evaluate Models
```bash
python src/train.py
```
*Executes 5-fold cross-validated hyperparameter tuning across Logistic Regression, Random Forest, and XGBoost, evaluates ROC-AUC/F1 metrics, performs SHAP explainability analysis, and saves model pipelines to `models/`.*

### 5. Launch Interactive Streamlit App
```bash
streamlit run app/app.py
```
Access the interactive web application at `http://localhost:8501`.

---

## 📊 Model Performance Benchmarks (Test Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Recommended)** | **0.8048** | **0.6189** | **0.6354** | **0.6270** | **0.8492** |
| **Random Forest Classifier** | 0.8013 | 0.6120 | 0.6215 | 0.6167 | 0.8431 |
| **Logistic Regression (Baseline)** | 0.7928 | 0.5891 | 0.6521 | 0.6190 | 0.8405 |

*Note: Models were trained using SMOTE rebalancing and tuned via 5-fold cross-validated GridSearchCV.*

---

## 💡 Key Business Insights

1. **Contract Commitment Impact:** Month-to-month contract holders experience ~42.7% churn versus <11% for 1-year/2-year contracts.
2. **Early Lifecycle Attrition:** Customers with <12 months tenure exhibit the highest churn vulnerability; proactive 30-60-90 day onboarding touchpoints are critical.
3. **High Monthly Charges & Fiber Optic Risk:** High monthly bills combined with Fiber Optic service correlate with higher churn due to competitive market pricing.
4. **Retention Anchors (Security & Tech Support):** Subscribing to Tech Support or Online Security reduces churn by over 50%.
5. **Electronic Check Friction:** Electronic check payment users experience nearly 45% churn rate, highlighting payment channel friction.

---

## 💻 Streamlit Web Application Features

1. **Single Customer Risk Assessment:** Input custom demographic & service details to obtain real-time churn probability, risk tier (**Low / Medium / High**), recommended action, and feature impact graph.
2. **Batch Customer CSV Prediction:** Drag-and-drop raw CSV files to batch-score thousands of customer accounts and export results as downloadable CSV files.
3. **Model & Insights Hub:** View real-time benchmarking metrics and strategic business recommendations.
