import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Disease Detector System", layout="wide")

@st.cache_data
def load_model(path="model.pkl"):
    return joblib.load(path)

def analyze_params(user_data):
    Preg, Glucose, BP, Skin, Insulin, BMI, DPF, Age = user_data
    lines = []

    if Glucose < 70:
        lines.append("Glucose is low — consider more slow carbs.")
    elif Glucose > 140:
        lines.append("Glucose is high — reduce sugary foods and consult a doctor.")
    else:
        lines.append("Glucose is in the safe range.")

    if BMI < 18.5:
        lines.append("Underweight — improve diet & increase protein.")
    elif BMI > 24.9:
        lines.append("BMI high — exercise and calorie control recommended.")
    else:
        lines.append("BMI in healthy range.")

    if BP > 80:
        lines.append("Blood pressure elevated — reduce salt & stress.")
    else:
        lines.append("Blood pressure normal.")

    if Insulin > 160:
        lines.append("High insulin — insulin resistance possible.")
    else:
        lines.append("Insulin seems normal.")

    return lines

def predict_with_model(model, arr):
    proba = model.predict_proba(arr)[0][1]
    pred = model.predict(arr)[0]
    return pred, proba


st.title("🚑 Disease Detector System — Advanced ML Analysis")
st.write("Enter your health details to generate a detailed risk analysis report.")

tabs = st.tabs(["📥 Input", "📊 Analysis", "📈 Charts"])

# ------------------------------------
# TAB 1 — INPUT
# ------------------------------------
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        Preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=90)
        BP = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72)
        Skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
        BMI = st.number_input("BMI", min_value=5.0, max_value=60.0, value=25.0, format="%.1f")
        DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5, format="%.3f")
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

    if st.button("Run Analysis"):
        st.session_state["run"] = True
        st.session_state["user"] = [Preg, Glucose, BP, Skin, Insulin, BMI, DPF, Age]

# ------------------------------------
# TAB 2 — ANALYSIS
# ------------------------------------
with tabs[1]:
    if "run" in st.session_state:
        user_data = st.session_state["user"]
        arr = np.array(user_data).reshape(1, -1)

        try:
            model = load_model("model.pkl")
            pred, proba = predict_with_model(model, arr)
            analysis = analyze_params(user_data)
        except:
            st.error("Model file not found! Train model first.")
        else:
            risk_type = "High Risk" if pred == 1 else "Low Risk"
            prob = round(proba * 100, 2)

            st.subheader("🔍 Risk Result")
            color = "red" if pred == 1 else "green"
            st.markdown(f"<h3 style='color:{color};'>{risk_type} ({prob}%)</h3>", unsafe_allow_html=True)

            st.subheader("📝 Detailed Parameter Analysis")
            for a in analysis:
                st.write("• " + a)

            st.subheader("📌 Summary")
            if pred == 1:
                st.warning("Your parameters indicate elevated risk. Follow strong precautions.")
            else:
                st.success("You are in the safe zone. Maintain your lifestyle.")

    else:
        st.info("Please go to Input tab and run analysis.")

# ------------------------------------
# TAB 3 — CHARTS
# ------------------------------------
with tabs[2]:
    if "run" in st.session_state:
        user = st.session_state["user"]
        labels = ["Glucose", "BMI", "BP"]
        values = [user[1], user[5], user[2]]

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=["blue", "orange", "green"])
        ax.set_title("Your Key Parameters")
        st.pyplot(fig)
    else:
        st.info("Run analysis first.")
