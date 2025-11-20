# 🚑 Disease Detector System (Machine Learning Project)

A Machine Learning based diabetes prediction system that gives:
- ✔ Risk Prediction (High / Low)
- ✔ Probability Score (in %)
- ✔ Parameter-wise Analysis (Glucose, BMI, BP etc.)
- ✔ Recommendations
- ✔ Streamlit UI with charts
- ✔ Data Visualizations

This project is made for the **Flipped Course – Introduction to Problem Solving & Programming**.

---

## 🔥 Features

- Predicts diabetes risk using Random Forest
- Provides detailed analysis of each health parameter
- Shows clean probability percentage
- Modern Streamlit UI with tabs
- Visual graphs for understanding dataset
- Modular Python code structure

---

## 📁 Project Structure

```

Disease-Detector-System/
│
├── main.py                # Terminal prediction
├── model.py               # Trains the ML model
├── predict.py             # Advanced analysis logic
├── streamlit_app.py       # Streamlit Web Interface
├── visualize.py           # Graphs & charts
├── requirements.txt       # Dependencies
├── data.csv               # Dataset
├── model.pkl              # Saved ML model
└── README.md              # Documentation

```

---

## 🧠 How It Works

### 1. **Training**
`model.py` trains a Random Forest Classifier on the diabetes dataset and generates `model.pkl`.

### 2. **Prediction**
`predict.py`:
- loads the ML model  
- predicts risk  
- generates probability  
- performs advanced analysis  

### 3. **UI**
`streamlit_app.py` creates a web interface with:
- Input Tab  
- Analysis Tab  
- Charts Tab  

---

## 📊 Visualizations

Using `visualize.py` you get:

- Glucose distribution  
- BMI distribution  
- Blood pressure histogram  
- Correlation heatmap  

These help understand the dataset better.

---

## 🚀 How to Run

### **Step 1 — Install Requirements**
```

pip install -r requirements.txt

```

### **Step 2 — Train Model**
```

python model.py

```

You will see:
```

Model saved as model.pkl

```

### **Step 3 — Run Terminal Prediction**
```

python main.py

```

### **Step 4 — Streamlit UI**
```

streamlit run streamlit_app.py

```

Open browser → `http://localhost:8501`

---

## 📝 Example Output

```

FINAL RESULT: High Risk
Probability: 74.21 %

DETAILED ANALYSIS:

* Glucose high — reduce sugary food
* BMI healthy
* BP normal
* Insulin high
* Age indicates moderate risk

SUMMARY:
Your health parameters show elevated risk factors.

```

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib  
- Seaborn  
- Joblib  

---

## 📌 Notes

This project is developed for academic learning purpose only  
and not intended for medical diagnosis.

---

## 👨‍💻 Author

**Govind Mewada**  
Reg. No: 25MIM10005  
Integrated M.Tech (AI), VIT Bhopal  
GitHub Username: `gsrcreation`

---