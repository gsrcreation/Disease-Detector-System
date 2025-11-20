import numpy as np
import joblib

def load_model():
    return joblib.load("model.pkl")


# ------------------------------------------
# ADVANCED PARAMETER-WISE ANALYSIS ENGINE
# ------------------------------------------

def parameter_analysis(name, value, normal_range, suggestions):
    """
    name = parameter name
    value = user value
    normal_range = (low, high)
    suggestions = dict containing 'low', 'high', 'okay'
    """

    low, high = normal_range

    if value < low:
        return f"{name} is low. {suggestions['low']}"
    elif value > high:
        return f"{name} is high. {suggestions['high']}"
    else:
        return f"{name} is in a healthy range. {suggestions['okay']}"


def calculate_analysis(user_data):
    analysis = []

    # Mapping user data indexes
    Preg, Glucose, BP, Skin, Insulin, BMI, DPF, Age = user_data

    analysis.append(parameter_analysis(
        "Glucose",
        Glucose,
        (70, 140),
        {
            "low": "You may experience weakness. Increase slow carbs.",
            "high": "High glucose indicates risk. Avoid sugar-heavy foods.",
            "okay": "Good glucose control."
        }
    ))

    analysis.append(parameter_analysis(
        "Blood Pressure",
        BP,
        (60, 80),
        {
            "low": "May cause dizziness. Increase hydration.",
            "high": "High BP increases heart and diabetes risk. Reduce salt.",
            "okay": "Blood pressure is stable."
        }
    ))

    analysis.append(parameter_analysis(
        "BMI",
        BMI,
        (18.5, 24.9),
        {
            "low": "Underweight. Improve protein intake.",
            "high": "High BMI indicates overweight. Exercise recommended.",
            "okay": "BMI is ideal."
        }
    ))

    analysis.append(parameter_analysis(
        "Insulin",
        Insulin,
        (15, 160),
        {
            "low": "Low insulin levels detected.",
            "high": "High insulin indicates resistance. Avoid sweets.",
            "okay": "Insulin level is balanced."
        }
    ))

    analysis.append(parameter_analysis(
        "Age",
        Age,
        (18, 45),
        {
            "low": "Teenage or young adult. Lower health risks.",
            "high": "Higher age increases diabetes probability.",
            "okay": "Age in moderate risk zone."
        }
    ))

    return analysis


# ------------------------------------------
# PREDICTION BLOCK
# ------------------------------------------

def predict(user_data):
    model = load_model()
    data = np.array(user_data).reshape(1, -1)

    probability = model.predict_proba(data)[0][1]
    result = model.predict(data)[0]

    detailed_analysis = calculate_analysis(user_data)

    output = {
        "risk": "High Risk" if result == 1 else "Low Risk",
        "probability": round(probability * 100, 2),
        "analysis": detailed_analysis,
        "summary": (
            "Your health parameters show elevated risk factors."
            if result == 1 else
            "Your overall health indicators are within safe range."
        )
    }

    return output
