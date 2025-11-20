# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Disease Detector System", layout="centered")

@st.cache_data
def load_model(path="model.pkl"):
    return joblib.load(path)

def analyze_params(user_data):
    # Reuse the advanced logic from predict.py for consistency
    Preg, Glucose, BP, Skin, Insulin, BMI, DPF, Age = user_data
    lines = []
    if Glucose < 70:
        lines.append("Glucose is low — consider more slow carbs.")
    elif Glucose > 140:
        lines.append("Glucose is high — reduce sugary foods and consult a doctor.")
    else:
        lines.append("Glucose in acceptable range.")
    if BMI < 18.5:
        lines.append("Underweight — increase protein intake.")
    elif BMI > 24.9:
        lines.append("Overweight — consider regular exercise and calorie control.")
    else:
        lines.append("BMI in healthy range.")
    if BP > 80:
        lines.append("Blood pressure high — watch salt intake and get BP checked.")
    else:
        lines.append("Blood pressure in acceptable range.")
    return lines

def predict_with_model(model, arr):
    proba = model.predict_proba(arr)[0][1]
    pred = model.predict(arr)[0]
    return pred, proba

st.title("Disease Detector System — Analysis v1")
st.markdown("Enter health parameters and get: probability, detailed parameter analysis, recommendations and simple charts.")

with st.expander("Example / Load sample dataset"):
    st.markdown("You can load a small sample dataset (Pima-style).")
    if st.button("Load sample data (for demo)"):
        try:
            sample = pd.read_csv("data.csv")
            st.dataframe(sample.head())
        except Exception as e:
            st.error("data.csv not found in project folder. Put your data.csv in the project folder.")
            st.write(e)

st.header("Patient input")
col1, col2 = st.columns(2)
with col1:
    Preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=90)
    BP = st.number_input("BloodPressure", min_value=0, max_value=200, value=72)
    Skin = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
with col2:
    Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
    BMI = st.number_input("BMI", min_value=5.0, max_value=60.0, value=25.0, format="%.1f")
    DPF = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, format="%.3f")
    Age = st.number_input("Age", min_value=1, max_value=120, value=30)

user_arr = np.array([Preg, Glucose, BP, Skin, Insulin, BMI, DPF, Age]).reshape(1, -1)

if st.button("Get analysis"):
    try:
        model = load_model("model.pkl")
    except Exception as e:
        st.error("Could not load model.pkl. Make sure model.pkl exists in project folder.")
        st.write(e)
    else:
        pred, proba = predict_with_model(model, user_arr)
        proba_pct = round(proba * 100, 2)
        st.metric("Risk probability", f"{proba_pct}%")
        st.subheader("Parameter-wise analysis")
        analysis = analyze_params([Preg, Glucose, BP, Skin, Insulin, BMI, DPF, Age])
        for a in analysis:
            st.write("- " + a)

        st.subheader("Recommendation")
        if pred == 1 or proba > 0.5:
            st.write("Potential risk detected. Suggest: consult a healthcare professional, reduce sugar intake, increase activity, and get clinical tests.")
        else:
            st.write("Low risk. Maintain healthy diet and activity; monitor periodically.")

        st.subheader("Quick charts")
        fig, ax = plt.subplots()
        ax.bar(["Glucose","BMI","BP"], [Glucose, BMI, BP])
        ax.set_ylabel("Value")
        st.pyplot(fig)

        st.info("This tool is for educational/demo purposes only — not medical advice.")
