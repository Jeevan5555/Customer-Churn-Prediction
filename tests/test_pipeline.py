import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_processing import DataCleaner, FeatureEngineer

class TestChurnPipeline(unittest.TestCase):
    
    def test_data_cleaner(self):
        cleaner = DataCleaner()
        df_raw = cleaner.load_raw_data()
        self.assertFalse(df_raw.empty, "Loaded dataframe should not be empty")
        
        df_clean = cleaner.clean_data(df_raw)
        self.assertNotIn('customerID', df_clean.columns, "customerID should be dropped")
        self.assertTrue(pd.api.types.is_numeric_dtype(df_clean['TotalCharges']), "TotalCharges should be numeric")
        self.assertTrue(set(df_clean['Churn'].unique()).issubset({0, 1}), "Churn target should be 0 or 1")

    def test_feature_engineer(self):
        cleaner = DataCleaner()
        df_raw = cleaner.load_raw_data()
        df_clean = cleaner.clean_data(df_raw)
        
        fe = FeatureEngineer()
        df_fe = fe.engineer_features(df_clean)
        
        expected_cols = ['TenureGroup', 'AvgMonthlySpend', 'TotalServices', 'HasSecurityBackup', 'HasStreamingService']
        for col in expected_cols:
            self.assertIn(col, df_fe.columns, f"Engineered feature {col} should exist")

    def test_model_inference(self):
        model_path = os.path.join(BASE_DIR, "models", "best_model.joblib")
        prep_path = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
        feat_path = os.path.join(BASE_DIR, "models", "feature_names.joblib")
        
        self.assertTrue(os.path.exists(model_path), "Model artifact best_model.joblib must exist")
        self.assertTrue(os.path.exists(prep_path), "Preprocessor artifact preprocessor.joblib must exist")
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        feature_names = joblib.load(feat_path)
        
        sample_path = os.path.join(BASE_DIR, "data", "sample_batch_customers.csv")
        sample_df = pd.read_csv(sample_path)
        if 'customerID' in sample_df.columns:
            sample_df = sample_df.drop(columns=['customerID'])
            
        fe = FeatureEngineer()
        sample_fe = fe.engineer_features(sample_df)
        
        X_proc = preprocessor.transform(sample_fe)
        X_proc_df = pd.DataFrame(X_proc, columns=feature_names)
        
        probs = model.predict_proba(X_proc_df)[:, 1]
        preds = model.predict(X_proc_df)
        
        self.assertEqual(len(probs), len(sample_df))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertTrue(set(preds).issubset({0, 1}))

if __name__ == '__main__':
    unittest.main()
