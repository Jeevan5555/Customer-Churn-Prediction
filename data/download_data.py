import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

PRIMARY_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp-for-data/master/data/Telco-Customer-Churn.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/WQU-MSc-QFIN/datasets/main/telco_churn.csv"

def download_or_generate_dataset(output_path=FILE_PATH):
    """
    Downloads the IBM Telco Customer Churn dataset from a public raw repository.
    If network access is unavailable, generates a synthetic dataset with identical
    schema, data types, and realistic churn relationships.
    """
    if os.path.exists(output_path):
        print(f"[INFO] Dataset already exists at: {output_path}")
        df = pd.read_csv(output_path)
        print(f"[INFO] Dataset shape: {df.shape}")
        return df

    urls_to_try = [PRIMARY_URL, FALLBACK_URL]
    download_success = False

    for url in urls_to_try:
        try:
            print(f"[INFO] Attempting download from: {url}")
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as response, open(output_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[SUCCESS] Dataset downloaded successfully to: {output_path}")
            download_success = True
            break
        except Exception as e:
            print(f"[WARNING] Could not download from {url}: {e}")

    if not download_success:
        print("[INFO] Generating synthetic Telco Customer Churn dataset...")
        df = generate_synthetic_telco_data()
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Synthetic dataset generated and saved to: {output_path}")
    else:
        df = pd.read_csv(output_path)

    print(f"[INFO] Dataset shape: {df.shape}")
    return df

def generate_synthetic_telco_data(n_samples=7043, random_state=42):
    """
    Generates a realistic Telco Customer Churn synthetic dataset matching
    the schema of Kaggle's WA_Fn-UseC_-Telco-Customer-Churn.csv.
    """
    np.random.seed(random_state)
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(n_samples)]
    gender = np.random.choice(['Female', 'Male'], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.30, 0.70])
    
    # Tenure in months (0 to 72)
    tenure = np.random.randint(0, 73, size=n_samples)
    
    phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.34, 0.44, 0.22])
    
    def get_add_on(internet):
        if internet == 'No':
            return 'No internet service'
        return np.random.choice(['Yes', 'No'], p=[0.40, 0.60])

    online_security = [get_add_on(is_srv) for is_srv in internet_service]
    online_backup = [get_add_on(is_srv) for is_srv in internet_service]
    device_protection = [get_add_on(is_srv) for is_srv in internet_service]
    tech_support = [get_add_on(is_srv) for is_srv in internet_service]
    streaming_tv = [get_add_on(is_srv) for is_srv in internet_service]
    streaming_movies = [get_add_on(is_srv) for is_srv in internet_service]
    
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], size=n_samples, p=[0.34, 0.23, 0.21, 0.22])
    
    # Monthly Charges calculation
    monthly_charges = []
    for i in range(n_samples):
        base = 20.0
        if phone_service[i] == 'Yes': base += 15.0
        if multiple_lines[i] == 'Yes': base += 10.0
        if internet_service[i] == 'DSL': base += 25.0
        elif internet_service[i] == 'Fiber optic': base += 45.0
        
        if online_security[i] == 'Yes': base += 8.0
        if online_backup[i] == 'Yes': base += 8.0
        if device_protection[i] == 'Yes': base += 8.0
        if tech_support[i] == 'Yes': base += 8.0
        if streaming_tv[i] == 'Yes': base += 10.0
        if streaming_movies[i] == 'Yes': base += 10.0
        
        # Add random noise
        base += np.random.normal(0, 3)
        monthly_charges.append(round(max(18.25, min(118.75, base)), 2))
        
    monthly_charges = np.array(monthly_charges)
    
    # Total Charges string (matching standard dataset where tenure=0 gives ' ')
    total_charges = []
    for t, m in zip(tenure, monthly_charges):
        if t == 0:
            total_charges.append(" ")
        else:
            tot = t * m + np.random.normal(0, 20)
            total_charges.append(str(round(max(m, tot), 2)))
            
    # Realistic Churn Probability calculation based on key domain rules:
    # High churn: Month-to-month, Fiber optic, low tenure, Electronic check, No TechSupport
    churn_prob = np.zeros(n_samples)
    for i in range(n_samples):
        score = 0.0
        if contract[i] == 'Month-to-month': score += 1.5
        elif contract[i] == 'One year': score -= 0.8
        elif contract[i] == 'Two year': score -= 1.8
        
        if tenure[i] < 12: score += 1.0
        elif tenure[i] > 48: score -= 1.2
        
        if internet_service[i] == 'Fiber optic': score += 0.8
        if tech_support[i] == 'No': score += 0.6
        if payment_method[i] == 'Electronic check': score += 0.5
        if monthly_charges[i] > 80: score += 0.4
        
        # Sigmoid function for probability
        prob = 1 / (1 + np.exp(-(score - 1.0)))
        churn_prob[i] = prob
        
    churn = np.where(churn_prob > np.percentile(churn_prob, 73.5), 'Yes', 'No')
    
    df = pd.DataFrame({
        'customerID': customer_ids,
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
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    return df

if __name__ == '__main__':
    download_or_generate_dataset()
