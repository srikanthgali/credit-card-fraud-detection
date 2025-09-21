import numpy as np
import pandas as pd
import joblib
import os
import sys

# Ensure the src directory is in the system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Preprocessor:
    def __init__(self):
        # Use RobustScaler for 'Amount' and StandardScaler for 'Time' as in EDA
        # Get the absolute path to the models directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
        print("Loading scaler from:", os.path.join(base_dir, "robust_scaler.joblib"))
        self.amount_scaler = joblib.load(os.path.join(base_dir, "robust_scaler.joblib"))
        self.time_scaler = joblib.load(os.path.join(base_dir, "standard_scaler.joblib"))

    def fit(self, X: pd.DataFrame):
        """
        Fit scalers to 'Amount' and 'Time' columns if present.
        """     
        if "Amount" in X.columns:
            self.amount_scaler.fit(X[["Amount"]])
        if "Time" in X.columns:
            self.time_scaler.fit(X[["Time"]])
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform 'Amount' and 'Time' columns using fitted scalers.
        Adds 'scaled_amount' and 'scaled_time' columns and drops originals.
        """
        X_copy = X.copy()
        if "Amount" in X_copy.columns:
            X_copy["scaled_amount"] = X_copy[["Amount"]]
            X_copy = X_copy.drop(columns=["Amount"])
        if "Time" in X_copy.columns:
            X_copy["scaled_time"] = X_copy[["Time"]]
            X_copy = X_copy.drop(columns=["Time"])
        return X_copy

    def fit_transform(self, X: pd.DataFrame):
        """
        Fit scalers and transform the data in one step.
        """
        return self.fit(X).transform(X)