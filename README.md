# Customer Churn Prediction Dashboard

A machine learning web dashboard that predicts whether a customer is likely to churn based on telecom customer data.  
The project includes data preprocessing, model training, evaluation, and an interactive dashboard for viewing predictions and model performance.

---

## Project Overview

Customer churn is one of the biggest problems for subscription-based businesses.  
This project helps identify customers who are likely to leave the service so that businesses can take preventive action.

The dashboard allows users to:

- Upload or use customer data
- Predict customer churn
- View model performance
- Analyze important churn factors
- Compare machine learning models
- Understand customer behavior using visualizations

---

## Problem Statement

Telecom companies lose revenue when customers discontinue their services.  
The goal of this project is to build a machine learning model that can predict whether a customer will churn based on customer information such as:

- Demographics
- Account details
- Service usage
- Contract type
- Payment method
- Monthly charges
- Tenure

---

## Objectives

The main objectives of this project are:

- Perform end-to-end data analysis
- Clean and preprocess customer churn data
- Train machine learning models
- Evaluate model performance using classification metrics
- Build an interactive dashboard
- Present predictions in a simple and user-friendly way

---

## Tech Stack

### Programming Language

- Python

### Libraries Used

- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- joblib

### Tools Used

- Git
- GitHub
- VS Code / Jupyter Notebook

---

## Project Features

- Customer churn prediction
- Interactive dashboard
- Model accuracy display
- Confusion matrix visualization
- Classification report
- Data insights and charts
- Clean user interface
- Saved machine learning model using joblib

---

## Machine Learning Workflow

The project follows a complete machine learning pipeline:

1. Data collection
2. Data cleaning
3. Exploratory Data Analysis
4. Feature encoding
5. Feature scaling
6. Train-test split
7. Model training
8. Model evaluation
9. Model saving
10. Dashboard deployment

---

## Models Used

The following machine learning models were used:

- Logistic Regression
- Random Forest Classifier

The models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## Project Structure
customer-churn-dashboard/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── model_results.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── images/
│   └── dashboard_preview.png
│
└── utils/
    └── preprocessing.py

Dataset Information

The dataset contains customer-level information used to predict churn.

Common Features
Customer ID
Gender
Senior Citizen
Partner
Dependents
Tenure
Phone Service
Internet Service
Contract
Payment Method
Monthly Charges
Total Charges
Churn
Target Variable
Column	Description
Churn	Indicates whether the customer left the service or not
Installation and Setup
1. Clone the Repository
git clone https://github.com/your-username/customer-churn-dashboard.git
2. Navigate to the Project Folder
cd customer-churn-dashboard
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

For Windows:

venv\Scripts\activate

For Mac/Linux:

source venv/bin/activate
5. Install Dependencies
pip install -r requirements.txt
6. Run the Streamlit App
streamlit run app.py
Usage

After running the app, open the local Streamlit URL in your browser.

Example:

http://localhost:8501

You can then:

View dashboard insights
Check model performance
Enter customer details
Predict whether a customer will churn
Model Performance
Metric	Score
Accuracy	89%
ROC-AUC	0.91
Model	Random Forest / Logistic Regression
Dashboard Preview

Add your dashboard screenshot here:

![Dashboard Preview](images/dashboard_preview.png)
Key Insights

Some possible insights from the churn analysis:

Customers with month-to-month contracts are more likely to churn.
Higher monthly charges may increase churn risk.
Customers with longer tenure are less likely to churn.
Electronic check payment method may be associated with higher churn.
Fiber optic internet users may show different churn behavior compared to DSL users.
Future Improvements

Possible improvements for this project:

Add more advanced models like XGBoost or LightGBM
Add model explainability using SHAP
Deploy the dashboard on Streamlit Cloud
Add real-time prediction using user input
Improve UI design
Add database integration
Add customer segmentation
Add downloadable prediction reports
