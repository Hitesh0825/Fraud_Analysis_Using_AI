import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("📌 Loading Dataset...")
df = pd.read_csv("AIML Dataset.csv")

# Extra features
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

# Drop columns not needed as per friend's code
df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

y = df_model["isFraud"]
X = df_model.drop("isFraud", axis=1)

categorical = ["type"]
numeric = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder="drop"
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

print("📌 Training Model (wait)...")
pipeline.fit(X_train, y_train)

print("\n📈 Evaluation Report:\n")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(pipeline, "fraud_analysis_pipeline.pkl")
print("\n✅ Model Saved as fraud_analysis_pipeline.pkl")
