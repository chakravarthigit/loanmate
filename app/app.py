"""
Hugging Face Deployment for LoanMate

This module provides a Gradio-based web interface for the LoanMate
loan approval prediction system.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import gradio as gr

from huggingface_hub import hf_hub_download

# Load the model from Hugging Face Hub
MODEL_REPO_ID = "Chakri5658/chakri-loanmate-model"
MODEL_FILENAME = "loan_master_model.pkl"
USE_MASTER_MODEL = False # Default to False
loan_master = None # Initialize loan_master

try:
    print(f"Downloading model from repo: {MODEL_REPO_ID}, file: {MODEL_FILENAME}")
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    print(f"Model downloaded to: {model_path}")
    loan_master = joblib.load(model_path)
    print("Loan Master Model loaded successfully using joblib.")
    USE_MASTER_MODEL = True
except Exception as e:
    print(f"Error loading Loan Master Model: {str(e)}")
    # sys.exit(f"Critical error: Model could not be loaded. {str(e)}") # Optional: exit if critical

def predict_loan_approval(loan_master, new_data, dataset_type='hugging'):
    """
    Make loan approval predictions using the master model.
    """
    if dataset_type not in loan_master['dataset_models']:
        raise ValueError(f"Dataset type '{dataset_type}' not found in model")
    
    # Get the relevant models and preprocessor
    model_info = loan_master['dataset_models'][dataset_type]
    preprocessor = model_info['preprocessor']
    ensemble = model_info['ensemble']
    
    # Preprocess the data
    X_processed = preprocessor.transform(new_data)
    
    # Make prediction
    prediction = ensemble.predict(X_processed)[0]
    probability = ensemble.predict_proba(X_processed)[0, 1]
    
    return prediction, probability

def calculate_risk_score(probability):
    """
    Calculate a risk score based on the prediction probability.
    """
    if probability >= 0.9:
        return 'Very Low Risk'
    elif probability >= 0.75:
        return 'Low Risk'
    elif probability >= 0.5:
        return 'Moderate Risk'
    elif probability >= 0.25:
        return 'High Risk'
    else:
        return 'Very High Risk'

def predict(income, credit_score, loan_amount, employment_status, age, loan_term, existing_debt, education_level):
    """
    Make a prediction for a loan application.
    """
    # Calculate derived features
    debt_to_income_ratio = (existing_debt / income) * 100 if income > 0 else 0
    loan_to_income_ratio = (loan_amount / income) * 100 if income > 0 else 0
    income_to_debt_ratio = income / (existing_debt + 1)
    
    # Create application data dictionary
    application_data = pd.DataFrame({
        'ApplicantIncome': [income],
        'CoapplicantIncome': [0],  # Default to 0 for simplicity
        'LoanAmount': [loan_amount / 1000],  # Convert to thousands as per model training
        'Loan_Amount_Term': [loan_term * 12],  # Convert years to months
        'Credit_History': [1 if credit_score >= 650 else 0],
        'Gender': ['Male'],  # Default value
        'Married': ['Yes'],  # Default value
        'Dependents': ['0'],  # Default value
        'Education': ['Graduate' if education_level in ['Bachelor', 'Master', 'PhD'] else 'Not Graduate'],
        'Self_Employed': ['Yes' if employment_status == 'Self-employed' else 'No'],
        'Property_Area': ['Urban']  # Default value
    })
    
    try:
        # Make prediction
        prediction, probability = predict_loan_approval(loan_master, application_data, dataset_type='hugging')
        
        # Calculate risk score
        risk_score = calculate_risk_score(probability)
        
        # Format results
        approval_status = "Approved" if prediction == 1 else "Rejected"
        approval_probability = f"{probability * 100:.2f}%"
        
        return f"Loan Status: {approval_status}\nApproval Probability: {approval_probability}\nRisk Assessment: {risk_score}"
    
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Annual Income ($)"),
        gr.Slider(300, 850, step=1, label="Credit Score"),
        gr.Number(label="Loan Amount ($)"),
        gr.Dropdown(["Full-time", "Part-time", "Self-employed", "Unemployed"], label="Employment Status"),
        gr.Number(label="Age"),
        gr.Slider(1, 30, step=1, label="Loan Term (years)"),
        gr.Number(label="Existing Debt ($)"),
        gr.Dropdown(["High School", "Associate", "Bachelor", "Master", "PhD"], label="Education Level")
    ],
    outputs="text",
    title="LoanMate - Loan Approval Prediction",
    description="Enter the loan application details to get a prediction.",
    examples=[
        [85000, 750, 250000, "Full-time", 35, 30, 45000, "Bachelor"],
        [42000, 630, 120000, "Part-time", 28, 15, 18000, "Associate"],
        [120000, 780, 350000, "Full-time", 45, 20, 65000, "Master"]
    ]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
else:
    # For Hugging Face Spaces
    demo.launch(share=False)