# LoanMate: Loan Approval Prediction System 🚀💸

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue?logo=huggingface&logoColor=yellow)](https://huggingface.co/spaces/Chakri5658/loanmate)

<div align="center">
  <img src="

![image/png](https://cdn-uploads.huggingface.co/production/uploads/67d84bd79d35da9f13ec9489/rf0zYp_TkRKW_oXaCn38v.png)
</div>

<p align="center">
  <a href="https://huggingface.co/spaces/Chakri5658/loanmate">
    <img alt="Hugging Face Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
  </a>
  <a href="https://huggingface.co/Chakri5658/chakri-loanmate-model">
    <img alt="Hugging Face Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow">
  </a>
  <a href="#">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Link-green">
  </a>
  <a href="#">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
  </a>
</p>

---

# LoanMate Model Details

This document provides a more in-depth look at the LoanMate loan approval prediction model.

## Model Architecture

The LoanMate model utilizes an **ensemble learning approach** to achieve robust and accurate predictions. It combines the strengths of several individual machine learning algorithms. The core idea is that by aggregating the predictions of multiple models, the overall performance can be improved, and the risk of relying on a single model's potential weaknesses is mitigated.

### Ensemble Components:

The ensemble typically includes the following types of models:

*   **Logistic Regression:** A linear model that's good for binary classification tasks and provides interpretable coefficients.
*   **Random Forest:** An ensemble of decision trees that handles non-linear relationships well and is robust to overfitting.
*   **Support Vector Machine (SVM):** Effective in high-dimensional spaces and can model complex decision boundaries.
*   **XGBoost (Extreme Gradient Boosting):** A highly efficient and scalable gradient boosting algorithm known for its performance in competitions.
*   **LightGBM (Light Gradient Boosting Machine):** Another gradient boosting framework that is fast and efficient, especially with large datasets.

Each of these models is trained on the preprocessed data, and their predictions are combined (e.g., through voting or averaging probabilities) to produce the final loan approval decision and probability score.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/67d84bd79d35da9f13ec9489/h8s8pe-MUtbywbJDnr2Jl.png)


## Datasets

The model was trained and evaluated on a combination of loan application datasets to ensure generalization across different applicant profiles and loan scenarios. 

Other internal or proprietary datasets might have also been used during the development of the model to enhance its robustness.

## Data Preprocessing

Consistent and thorough data preprocessing is crucial for the performance of any machine learning model. The following steps are applied to the raw input data before it's fed to the models:

1.  **Handling Missing Values:**
    *   Numerical features: Missing values are imputed using the mean or median of the respective column.
    *   Categorical features: Missing values are imputed using the most frequent value (mode) of the respective column.

2.  **Feature Engineering (Derived Features):
    *   `DebtToIncomeRatio`: Calculated as `(Existing Debt / Annual Income) * 100`.
    *   `LoanToIncomeRatio`: Calculated as `(Loan Amount / Annual Income) * 100`.
    *   `IncomeToDebtRatio`: Calculated as `Annual Income / (Existing Debt + 1)` (1 is added to avoid division by zero).
    *   Credit History is often binarized (e.g., 1 for good history, 0 for bad).

3.  **Encoding Categorical Features:**
    *   **One-Hot Encoding:** Categorical features with no inherent order (nominal features like 'Gender', 'Married', 'Self_Employed', 'Property_Area') are converted into numerical format using one-hot encoding. This creates new binary columns for each category.
    *   **Ordinal Encoding/Mapping:** Categorical features with an inherent order (ordinal features like 'Education', 'Dependents') might be mapped to numerical values if not handled by one-hot encoding.

4.  **Scaling Numerical Features:**
    *   **StandardScaler:** Numerical features (like 'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term') are scaled to have zero mean and unit variance. This helps algorithms that are sensitive to feature magnitudes (e.g., SVM, Logistic Regression).

These preprocessing steps are encapsulated within a `ColumnTransformer` and `Pipeline` from `scikit-learn`. This ensures that the same transformations are applied consistently during training, evaluation, and prediction.

## Model Training and Saving

The `build_master_model` function in `loan_master_model.py` is responsible for orchestrating the training process for different dataset types (like 'hugging' and 'loan_approval'). For each dataset type, it defines the specific preprocessor and trains the ensemble model.

The final trained model, which includes the preprocessors and the ensemble models for each dataset type, is saved as a single `loan_master_model.pkl` file using `joblib`. `joblib` is preferred over `pickle` for saving scikit-learn models as it's more efficient with large NumPy arrays.

## Prediction

When a new loan application is submitted via the Gradio interface:

1.  The input data is converted into a Pandas DataFrame.
2.  Derived features are calculated.
3.  The `predict_loan_approval` function is called.
4.  This function selects the appropriate preprocessor and ensemble model based on the `dataset_type` (which is 'hugging' for the Space).
5.  The data is transformed using the loaded preprocessor.
6.  The ensemble model makes a prediction (0 for rejected, 1 for approved) and calculates the probability of approval.
7.  A risk score is then derived from this probability.

This detailed process ensures that predictions are made consistently with how the model was trained.
