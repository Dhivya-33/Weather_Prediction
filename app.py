import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🌦️ Weather Prediction App (Rain Tomorrow)")
st.write("Predict whether it will rain tomorrow based on today's weather conditions.")

uploaded_file = st.file_uploader("Upload weatherAUS.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    
    features = [
        "MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm", "WindSpeed9am", "Temp9am", "Temp3pm"
    ]

    df = df[features + ["RainTomorrow"]]

    
    df = df.dropna()

    
    le = LabelEncoder()
    df["RainTomorrow"] = le.fit_transform(df["RainTomorrow"])

    X = df[features]
    y = df["RainTomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy: **{acc*100:.2f}%**")

    st.subheader("🔍 Enter Today's Weather Details")

    user_input = {}

    for col in features:
        user_input[col] = st.number_input(f"{col}", value=0.0)

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Rain Tomorrow"):
        prediction = model.predict(input_df)[0]
        result = "YES, it will rain tomorrow." if prediction == 1 else " NO, it will not rain tomorrow."

        st.subheader("Prediction Result")
        st.write(result)
