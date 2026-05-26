import streamlit as st
import pandas as pd
import pickle
import numpy as np

# LOAD MODEL

with open("models/knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# LOAD SCALER

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# PAGE TITLE

st.title("Heart Disease Prediction")

st.write("KNN Classifier Machine Learning App")

# USER INPUTS

age = st.slider("Age", 20, 80, 40)

sex = st.selectbox("Sex", [0, 1])

cp = st.slider("Chest Pain Type", 0, 3, 1)

trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)

chol = st.slider("Cholesterol", 100, 600, 200)

thalach = st.slider("Maximum Heart Rate", 60, 220, 150)

# PREDICTION BUTTON

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, thalach]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")
