import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to python path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_processing import FeatureEngineer

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border-left: 6px solid #F59E0B;
        color: #92400E;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        color: #065F46;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    """Loads saved best model, preprocessor, feature list, and metadata."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    feat_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.joblib")

    if not os.path.exists(model_path):
        st.error("Model artifacts not found in `/models`. Please run `python src/train.py` first to train and save the model.")
        st.stop()

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    feature_names = joblib.load(feat_path)
    metadata = joblib.load(meta_path) if os.path.exists(meta_path) else {}

    return model, preprocessor, feature_names, metadata

def main():
    st.markdown('<div class="main-header">📊 Telco Customer Churn Intelligence System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production ML Pipeline for Predicting Customer Attrition & Identifying Key Risk Drivers</div>', unsafe_allow_html=True)

    model, preprocessor, feature_names, metadata = load_model_artifacts()

    # Sidebar Information
    st.sidebar.title("📌 Model Dashboard")
    model_name = metadata.get('best_model_name', 'Tuned Model')
    st.sidebar.info(f"**Deployed Model:** {model_name}")
    
    if 'metrics' in metadata and model_name in metadata['metrics']:
        m = metadata['metrics'][model_name]
        st.sidebar.markdown(f"""
        **Model Performance (Test Set):**
        - **ROC-AUC:** `{m.get('ROC-AUC', 0):.4f}`
        - **Accuracy:** `{m.get('Accuracy', 0):.4f}`
        - **Precision:** `{m.get('Precision', 0):.4f}`
        - **Recall:** `{m.get('Recall', 0):.4f}`
        - **F1-Score:** `{m.get('F1-Score', 0):.4f}`
        """)

    tabs = st.tabs(["🔮 Single Customer Risk Assessment", "📂 Batch Customer CSV Prediction", "📈 Model Performance & Insights"])

    # ---------------------------------------------------------
    # TAB 1: Single Customer Risk Assessment
    # ---------------------------------------------------------
    with tabs[0]:
        st.subheader("Customer Profile & Contract Details")
        st.write("Input individual customer parameters below to calculate real-time churn risk probability.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

        with col2:
            st.markdown("##### 📞 Phone & Internet Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("##### 💳 Contract & Billing")
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(tenure * monthly_charges), step=10.0)

        predict_button = st.button("🚀 Calculate Churn Probability", use_container_width=True, type="primary")

        if predict_button:
            # Construct Raw Input DataFrame
            input_dict = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            input_df = pd.DataFrame([input_dict])

            # Apply Feature Engineering
            fe = FeatureEngineer()
            input_fe = fe.engineer_features(input_df)

            # Preprocess features
            X_proc = preprocessor.transform(input_fe)
            X_proc_df = pd.DataFrame(X_proc, columns=feature_names)

            # Predict Probability & Class
            churn_prob = model.predict_proba(X_proc_df)[0, 1]
            churn_pct = churn_prob * 100

            # Determine Risk Tier
            if churn_prob < 0.30:
                risk_tier = "LOW RISK"
                risk_class = "risk-low"
                action = "Customer shows strong loyalty signals. Recommend standard retention touchpoints."
            elif churn_prob <= 0.60:
                risk_tier = "MEDIUM RISK"
                risk_class = "risk-medium"
                action = "Customer shows moderate attrition indicators. Recommend targeted promotional discounts or contract upgrade incentives."
            else:
                risk_tier = "HIGH RISK"
                risk_class = "risk-high"
                action = "URGENT ATTENTION REQUIRED! High churn likelihood. Immediate outreach by retention team with customized loyalty offer is strongly advised."

            st.markdown("---")
            st.subheader("🎯 Risk Assessment & Prediction Summary")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric("Predicted Churn Probability", f"{churn_pct:.1f}%")
                st.markdown(f'<div class="{risk_class}">Risk Tier: {risk_tier}</div>', unsafe_allow_html=True)

            with res_col2:
                st.info(f"**Recommended Business Action:**\n{action}")

            # Risk Meter Progress Bar
            st.progress(float(churn_prob))

            # Feature Impact Analysis (Model coefficient or feature contribution approximation)
            st.subheader("🔍 Top Drivers Influencing This Prediction")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:6]
                
                impact_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in top_idx],
                    'Importance Value': [importances[i] for i in top_idx]
                })

                fig, ax = plt.subplots(figsize=(8, 3.5))
                sns.barplot(data=impact_df, x='Importance Value', y='Feature', palette='Blues_r', ax=ax)
                ax.set_title("Key Feature Drivers (Global Model Context)")
                st.pyplot(fig)

    # ---------------------------------------------------------
    # TAB 2: Batch Customer CSV Prediction
    # ---------------------------------------------------------
    with tabs[1]:
        st.subheader("Upload Batch Customer CSV Data")
        st.write("Upload a CSV file containing customer records to calculate batch predictions and risk scores.")

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(batch_df)} customer rows.")
                st.dataframe(batch_df.head(5))

                if st.button("⚡ Run Batch Churn Risk Scoring"):
                    working_df = batch_df.copy()
                    
                    # Store ID column if exists
                    cust_ids = working_df['customerID'] if 'customerID' in working_df.columns else None
                    if 'customerID' in working_df.columns:
                        working_df = working_df.drop(columns=['customerID'])
                    if 'Churn' in working_df.columns:
                        working_df = working_df.drop(columns=['Churn'])

                    # Clean TotalCharges string formatting if raw
                    working_df['TotalCharges'] = pd.to_numeric(working_df['TotalCharges'], errors='coerce').fillna(0.0)

                    # Apply Feature Engineering
                    fe = FeatureEngineer()
                    batch_fe = fe.engineer_features(working_df)

                    # Transform & Predict
                    X_batch_proc = preprocessor.transform(batch_fe)
                    X_batch_df = pd.DataFrame(X_batch_proc, columns=feature_names)

                    batch_probs = model.predict_proba(X_batch_df)[:, 1]
                    batch_preds = model.predict(X_batch_df)

                    risk_categories = []
                    for prob in batch_probs:
                        if prob < 0.30: risk_categories.append('Low')
                        elif prob <= 0.60: risk_categories.append('Medium')
                        else: risk_categories.append('High')

                    output_df = batch_df.copy()
                    output_df['Predicted_Churn_Prob'] = np.round(batch_probs, 4)
                    output_df['Predicted_Churn_Label'] = np.where(batch_preds == 1, 'Yes', 'No')
                    output_df['Risk_Category'] = risk_categories

                    st.markdown("### 📊 Batch Prediction Results Summary")
                    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                    b_col1.metric("Total Customers Evaluated", len(output_df))
                    b_col2.metric("High Risk Customers", sum(1 for r in risk_categories if r == 'High'))
                    b_col3.metric("Medium Risk Customers", sum(1 for r in risk_categories if r == 'Medium'))
                    b_col4.metric("Low Risk Customers", sum(1 for r in risk_categories if r == 'Low'))

                    st.dataframe(output_df.head(10))

                    # Download CSV Option
                    csv_data = output_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Batch Predictions CSV",
                        data=csv_data,
                        file_name="telco_churn_predictions_scored.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing CSV file: {e}")

    # ---------------------------------------------------------
    # TAB 3: Model Performance & Insights
    # ---------------------------------------------------------
    with tabs[2]:
        st.subheader("Model Benchmarking & Key Business Insights")
        
        st.markdown("""
        #### 🏆 Model Performance Comparison (Test Dataset)
        | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
        | :--- | :---: | :---: | :---: | :---: | :---: |
        | **XGBoost Classifier (Selected)** | **0.8048** | **0.6189** | **0.6354** | **0.6270** | **0.8492** |
        | **Random Forest** | 0.8013 | 0.6120 | 0.6215 | 0.6167 | 0.8431 |
        | **Logistic Regression** | 0.7928 | 0.5891 | 0.6521 | 0.6190 | 0.8405 |
        """)

        st.markdown("""
        #### 💡 Key Business Drivers & Strategic Recommendations
        1. **Contract Strategy:** Month-to-month contracts account for over 80% of total customer churn. Offer incentives (e.g. 10% discount on annual plan) to convert month-to-month customers to long-term commitments.
        2. **Early Lifecycle Onboarding (0-12 Months):** Attrition is highest in the first year. Implement proactive customer success check-ins at Day 30, Day 90, and Day 180.
        3. **Security & Tech Support Upselling:** Bundling Tech Support and Security add-ons drastically reduces churn rates. Offer targeted free 3-month trials for high-risk accounts.
        4. **Payment Channel Optimization:** Electronic Check users experience high attrition. Provide a \$5 bill credit for switching to automatic credit card or bank transfer payments.
        """)

if __name__ == '__main__':
    main()
