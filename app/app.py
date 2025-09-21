import sys
import os
import streamlit as st
import pandas as pd
import json

# Ensure the src directory is in the system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_loader import load_model
from src.data_preprocessing import Preprocessor

# Load the trained XGBoost model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/xgboost_smote_model.joblib"))
model = load_model(MODEL_PATH)

# Load example transactions
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
example_files = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json")]

examples = {}
for file in example_files:
    with open(os.path.join(EXAMPLES_DIR, file), "r") as f:
        examples[file] = json.load(f)

# Page title and description
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details manually or select an example to predict Fraud or Not Fraud.")

# Sidebar input form for user to enter transaction features
st.sidebar.header("Transaction Features")

# Option to select an example transaction
example_choice = st.sidebar.selectbox(
    "Choose an example transaction (or select 'Manual Input')",
    options=["Manual Input"] + list(examples.keys())
)

# Initialize input data
if example_choice != "Manual Input":
    # Load selected example
    input_data = examples[example_choice]
    st.sidebar.write(f"Loaded example: {example_choice}")
else:
    # Manual input for transaction time and amount
    time = st.sidebar.number_input("Transaction Time (seconds)", min_value=0.0, max_value=200000.0, value=1000.0, format="%.6f")
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=50.0, format="%.6f")

    # Inputs for anonymized PCA features V1–V28
    features = {}
    for i in range(1, 29):
        features[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0, format="%.6f")

    # Collect all inputs into a dictionary
    input_data = {
        "Time": [time],
        "Amount": [amount],
    }
    input_data.update({k: [v] for k, v in features.items()})

# Ensure all values in input_data are lists
input_data = {k: v if isinstance(v, list) else [v] for k, v in input_data.items()}

# Convert input data to DataFrame
df_input = pd.DataFrame(input_data)

# Preprocess the input data (scaling, etc.)
preprocessor = Preprocessor()
df_input = preprocessor.fit_transform(df_input)

# Make prediction when the user clicks the button
if st.button("Predict Fraud"):
    # Predict class (0: Not Fraud, 1: Fraud)
    prediction = model.predict(df_input)[0]
    # Predict probability of fraud
    prediction_proba = model.predict_proba(df_input)[0][1]

    # Display result to user
    if prediction == 1:            
        st.error(f"⚠️ Transaction is FRAUDULENT (Prob: {prediction_proba:.6f})")
    else:
        st.success(f"✅ Transaction is NOT Fraud (Prob: {prediction_proba:.6f})")
    st.write("### Transaction Details")
    st.json(input_data)  # Display the input data in JSON format for user reference