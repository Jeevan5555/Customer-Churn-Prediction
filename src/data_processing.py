import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class DataCleaner:
    """
    Handles data loading, cleaning, target encoding, and basic validation.
    """
    def __init__(self, filepath=None):
        if filepath is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        self.filepath = filepath

    def load_raw_data(self):
        """Loads dataset and prints basic structural summary."""
        print(f"[DATA LOAD] Loading data from {self.filepath}...")
        df = pd.read_csv(self.filepath)
        print(f"[DATA LOAD] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"[DATA LOAD] Duplicates: {df.duplicated().sum()}")
        print("\n--- Raw Data Schema & Sample ---")
        print(df.info())
        return df

    def clean_data(self, df):
        """
        Cleans raw dataset:
        1. Fixes TotalCharges string data type issue.
        2. Imputes missing TotalCharges values.
        3. Encodes target variable Churn (Yes=1, No=0).
        """
        df_clean = df.copy()
        
        # Drop customerID if present
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop(columns=['customerID'])
            
        # Fix TotalCharges data type issue (spaces ' ' converted to NaN)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Missing values strategy:
        # TotalCharges has NaNs where tenure is 0. Since no billing cycles have completed,
        # fill missing TotalCharges with 0.0 or the median.
        missing_tc = df_clean['TotalCharges'].isna().sum()
        if missing_tc > 0:
            print(f"[DATA CLEAN] Imputing {missing_tc} missing TotalCharges values with 0.0 (tenure=0).")
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
            
        # Encode Target variable: Churn -> 1 / 0
        if 'Churn' in df_clean.columns:
            df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).astype(int)
            print(f"[DATA CLEAN] Churn target encoded. Distribution:\n{df_clean['Churn'].value_counts(normalize=True)}")
            
        return df_clean

class FeatureEngineer:
    """
    Generates domain-specific derived features for customer churn analysis.
    """
    def __init__(self):
        pass

    def engineer_features(self, df):
        """
        Creates derived features:
        - TenureGroup: Binned tenure periods
        - AvgMonthlySpend: Historical average monthly charge (TotalCharges / (tenure + 1))
        - TotalServices: Count of total subscribed services
        - HasSecurityBackup: Flag for security + backup add-ons
        - HasStreamingService: Flag for TV or Movies streaming
        """
        df_fe = df.copy()
        
        # 1. Tenure Grouping
        bins = [-1, 12, 24, 48, 60, 100]
        labels = ['0-12 Mo', '12-24 Mo', '24-48 Mo', '48-60 Mo', '60+ Mo']
        df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], bins=bins, labels=labels)
        
        # 2. Average Monthly Spend (derived ratio)
        df_fe['AvgMonthlySpend'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1.0)
        
        # 3. Count of Total Subscribed Services
        service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        def count_services(row):
            count = 0
            for col in service_cols:
                if col in row and row[col] == 'Yes':
                    count += 1
            return count

        df_fe['TotalServices'] = df_fe.apply(count_services, axis=1)
        
        # 4. Security & Backup Bundle Flag
        df_fe['HasSecurityBackup'] = (
            ((df_fe['OnlineSecurity'] == 'Yes') & (df_fe['OnlineBackup'] == 'Yes')).astype(int)
        )
        
        # 5. Streaming Bundle Flag
        df_fe['HasStreamingService'] = (
            ((df_fe['StreamingTV'] == 'Yes') | (df_fe['StreamingMovies'] == 'Yes')).astype(int)
        )

        print(f"[FEATURE ENG] Engineered {5} new domain features. Total columns: {df_fe.shape[1]}")
        return df_fe

class PreprocessingPipeline:
    """
    Handles encoding, feature scaling, SMOTE class balancing, and Train-Test split.
    """
    def __init__(self, target_col='Churn', test_size=0.2, random_state=42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None

    def prepare_data(self, df, apply_smote=True):
        """
        Prepares X, y, splits data 80/20, applies One-Hot Encoding, StandardScaler,
        and SMOTE oversampling on training data.
        """
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Define categorical vs numerical features
        num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"[PIPELINE] Numerical features ({len(num_features)}): {num_features}")
        print(f"[PIPELINE] Categorical features ({len(cat_features)}): {cat_features}")

        # ColumnTransformer with StandardScaler and OneHotEncoder
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
            ]
        )

        # Stratified Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"[PIPELINE] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Fit preprocessor on training data and transform train & test
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # Retrieve feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
        self.feature_names = num_features + encoded_cat_names

        X_train_df = pd.DataFrame(X_train_proc, columns=self.feature_names, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_proc, columns=self.feature_names, index=X_test.index)

        # Apply SMOTE to training set if requested
        if apply_smote:
            print("[PIPELINE] Applying SMOTE to rebalance training data...")
            smote = SMOTE(random_state=self.random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train_df, y_train)
            print(f"[PIPELINE] Post-SMOTE Train shape: {X_train_res.shape}, Class Balance: {np.bincount(y_train_res)}")
            return X_train_res, X_test_df, y_train_res, y_test, X_train, X_test
        
        return X_train_df, X_test_df, y_train, y_test, X_train, X_test

def get_processed_data():
    """Utility function to load, clean, engineer, and return train/test datasets."""
    cleaner = DataCleaner()
    df_raw = cleaner.load_raw_data()
    df_clean = cleaner.clean_data(df_raw)
    
    fe = FeatureEngineer()
    df_fe = fe.engineer_features(df_clean)
    
    pipeline = PreprocessingPipeline()
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = pipeline.prepare_data(df_fe)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'pipeline': pipeline,
        'df_clean': df_clean,
        'df_fe': df_fe
    }

if __name__ == '__main__':
    data = get_processed_data()
    print("\n[SUCCESS] Data processing pipeline executed successfully!")
