# LoanMate: Loan Approval Prediction System 🚀💸

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-HuggingFace-blue?logo=huggingface&logoColor=yellow)](https://huggingface.co/spaces/Chakri5658/loanmate)

![LoanMate Banner](https://img.icons8.com/color/96/000000/bank.png)

## Overview

---

## 🌐 Live Demo

Try LoanMate instantly on Hugging Face Spaces:

[![Open in Spaces](https://img.shields.io/badge/Open%20in%20HuggingFace%20Spaces-%F0%9F%A4%97-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Chakri5658/loanmate)

---

**LoanMate** is an advanced, end-to-end machine learning solution for predicting loan approvals. It features robust data preprocessing, multiple ML models (including an ensemble master model), a user-friendly web interface, batch processing, and insightful visualizations. Built for flexibility and accuracy, LoanMate is ideal for financial institutions, data scientists, and developers.

---

## ✨ Features

- 🔥 **Ensemble Model**: Combines multiple algorithms for superior prediction accuracy
- 🧹 **Automated Data Preprocessing**: Cleans, transforms, and engineers features
- 🧠 **Multiple ML Models**: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
- 📊 **Performance Visualizations**: Confusion matrix, ROC curve, feature importance, and more
- 🌐 **Web UI**: Flask-based interface for single & batch predictions
- 📁 **Batch Processing**: Upload CSVs for bulk predictions
- 🛠️ **API Access**: REST endpoint for programmatic predictions
- 🏦 **Realistic Features**: Handles income, credit score, assets, debts, and more
- 🏷️ **Risk Scoring**: Categorizes applications by risk level

---

## 🗂️ Project Structure

```
loanmate_model/
├── huggingface_deployment/   # Deployment scripts for HuggingFace
├── loan_master_model/        # Saved ensemble model & visualizations
├── models/                   # Trained models, preprocessors, feature names
├── src/                      # Core ML pipeline (preprocessing, training, prediction)
├── ui/                       # Flask web app (UI, static, templates)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/loanmate_model.git
cd loanmate_model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Models
```bash
# Make sure your data is in the correct format, then run:
python src/model_training.py
```

### 4. Launch the Web App
```bash
python ui/app.py
```
Visit [http://localhost:5001](http://localhost:5001) in your browser.

---

## 🖥️ Web Interface

- **Single Prediction:** Fill out the form and get instant results with risk score and confidence.
- **Batch Prediction:** Upload a CSV of applications for bulk results and download the output.
- **Dashboard:** Visualize model performance, feature importances, and more.

![UI Icon](https://img.icons8.com/color/48/000000/combo-chart--v2.png)

---

## 🧩 Core Modules

- `src/data_preprocessing.py`  
  Data cleaning, feature engineering, and transformation.
- `src/model_training.py`  
  Trains and evaluates multiple ML models, saves best models and visualizations.
- `src/loan_master_prediction.py`  
  Ensemble prediction logic, supports multiple data formats and batch processing.
- `ui/app.py`  
  Flask web app for user interaction and API access.

---

## 📦 Deployment

- **HuggingFace Spaces:** Scripts and configs in `huggingface_deployment/` for cloud deployment.
- **API:** Use `/api/predict` endpoint for programmatic access.

---

## 📊 Example Visualizations

- Confusion Matrix
- ROC Curve
- Feature Importance
- Model Comparison

![Confusion Matrix](loan_master_model/hugging_confusion_matrix.png)
![ROC Curve](loan_master_model/hugging_roc_curve.png)

---

## 📝 Sample Prediction (Python)

```python
from src.loan_master_prediction import predict_with_loan_master

sample_application = {
    'ApplicantIncome': 5000,
    'CoapplicantIncome': 1000,
    'LoanAmount': 200,
    'Loan_Amount_Term': 360,
    'Gender': 'Male',
    'Married': 'Yes',
    'Dependents': '0',
    'Education': 'Graduate',
    'Self_Employed': 'No',
    'Credit_History': 1.0,
    'Property_Area': 'Urban'
}

result = predict_with_loan_master(sample_application, dataset_type='hugging')
print(result)
```

---

## 🛡️ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

- Built with ❤️ by Siva Nagaraju and contributors
- Powered by scikit-learn, XGBoost, LightGBM, Flask, Plotly, and more

---

## 📬 Contact

For questions, suggestions, or contributions:
- GitHub Issues
- Email: your.email@example.com

---

> "Empowering smarter lending decisions with AI!" 🤖🏦
