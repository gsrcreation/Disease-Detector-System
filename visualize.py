import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data.csv")

# 1. Glucose Distribution
plt.figure(figsize=(8,4))
sns.histplot(data['Glucose'], kde=True)
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose")
plt.ylabel("Count")
plt.show()

# 2. BMI Distribution
plt.figure(figsize=(8,4))
sns.histplot(data['BMI'], kde=True, color='orange')
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.show()

# 3. BloodPressure Distribution
plt.figure(figsize=(8,4))
sns.histplot(data['BloodPressure'], kde=True, color='green')
plt.title("Blood Pressure Distribution")
plt.xlabel("Blood Pressure")
plt.ylabel("Count")
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
