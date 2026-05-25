import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("Customer Churn Prediction Dashboard")
st.write("This dashboard predicts customer churn and shows simple business insights.")

if not os.path.exists("churn_model.pkl"):
    st.error("Model file not found. Please run: python model_training.py")
    st.stop()

if not os.path.exists("model_results.csv"):
    st.error("Model results file not found. Please run: python model_training.py")
    st.stop()

df = pd.read_csv("data/telco_churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
results_df = pd.read_csv("model_results.csv", index_col=0)

tab1, tab2, tab3 = st.tabs(["Business Overview", "Model Performance", "Predict Churn"])

with tab1:
    st.subheader("Customer Overview")

    col1, col2, col3 = st.columns(3)

    total_customers = df.shape[0]
    churn_rate = df[df["Churn"] == "Yes"].shape[0] / total_customers * 100
    avg_monthly = df["MonthlyCharges"].mean()

    col1.metric("Total Customers", total_customers)
    col2.metric("Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

    fig1 = px.pie(df, names="Churn", title="Churn Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="tenure", color="Churn", title="Tenure vs Churn")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x="Churn", y="MonthlyCharges", title="Monthly Charges by Churn")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Model Evaluation")

    st.dataframe(results_df)

    fig4 = px.bar(
        results_df,
        x=results_df.index,
        y="F1 Score",
        title="Model Comparison by F1 Score"
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Predict Customer Churn")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure Months", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone],
        "InternetService": [internet],
        "Contract": [contract],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    input_encoded = pd.get_dummies(input_data)

    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_columns]

if st.button("Predict Churn"):
        prediction = model.predict(input_encoded)[0]

        if hasattr(model, "predict_proba"):
            try:
                probability = model.predict_proba(input_encoded)[0][1]
            except:
                probability = prediction
        else:
            probability = prediction

        if prediction == 1:
            st.error(f"Customer is likely to churn. Risk: {float(probability):.2%}")
        else:
            st.success(f"Customer is likely to stay. Churn risk: {float(probability):.2%}")