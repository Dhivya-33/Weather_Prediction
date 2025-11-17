
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("weatherAUS.csv")

features = [
    "MinTemp", "MaxTemp", "Rainfall", "Humidity9am",
    "Humidity3pm", "Pressure9am", "Pressure3pm",
    "WindSpeed9am", "Temp9am", "Temp3pm"
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

print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix - Weather Prediction (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test, y_pred))
