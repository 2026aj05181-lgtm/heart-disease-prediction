
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

st.title("Heart Disease Prediction")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    models = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "SVM": "svm.pkl"
    }

    model = joblib.load(models[model_name])

    predictions = model.predict(data)

    st.write("Predictions")
    st.write(predictions)
