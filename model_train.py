# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==============================
# 2. MLflow Config
# ==============================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "casting_defect_model_v2"
MLFLOW_ARTIFACT_LOCATION = "./mlartifacts"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ==============================
# 3. Load Dataset
# ==============================
# Replace this with your dataset
df = pd.DataFrame({
    "age": [25, 32, 47, 51, 62, 23, 34, 45, 52, 46],
    "salary": [50000, 60000, 80000, 81000, 90000, 45000, 70000, 85000, 91000, 79000],
    "purchased": [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
})

X = df[["age", "salary"]]
y = df["purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ==============================
# 4. Train Logistic Regression
# ==============================
model = LogisticRegression()
model.fit(X_train, y_train)


# ==============================
# 5. Log to MLflow
# ==============================
with mlflow.start_run(run_name="logreg_model_v1"):

    # ✅ Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("solver", model.solver)

    # ✅ Log metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # ✅ Log dataset file (for reference)
    df.to_csv("input_data.csv", index=False)
    mlflow.log_artifact("input_data.csv", artifact_path="data")

    # ✅ Log trained Logistic Regression model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logreg_model"
    )

    print(f"✅ Logistic Regression model logged with accuracy: {acc:.4f}")
