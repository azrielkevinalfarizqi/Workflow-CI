import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import shutil

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    # =====================
    # PATH & DATA LOAD
    # =====================
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "Metro_Interstate_Traffic_Volume_preprocessing.csv")
    data = pd.read_csv(csv_path)

    # =====================
    # FEATURE ENGINEERING
    # =====================
    data["date_time"] = pd.to_datetime(data["date_time"])
    data["hour"] = data["date_time"].dt.hour
    data["dayofweek"] = data["date_time"].dt.dayofweek
    data = data.drop(columns=["date_time"])

    X = data.drop(columns=["traffic_volume"])
    y = data["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =====================
    # MLflow SETUP (FRESH)
    # =====================
    mlruns_path = os.path.join(base_dir, "mlruns")
    if os.path.exists(mlruns_path):
        shutil.rmtree(mlruns_path)  # Hapus mlruns lama agar CI fresh
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    mlflow.set_experiment("Traffic Volume Regression - Baseline")

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)

    # =====================
    # DEFINE MODELS
    # =====================
    models = {
        "RandomForest_Baseline": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "SVR_Baseline": SVR(
            kernel="rbf",
            C=100,
            gamma=0.1,
            epsilon=0.1
        )
    }

    # =====================
    # TRAIN & LOG
    # =====================
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Log hyperparameters
            mlflow.log_params(model.get_params())

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log metrics
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # Plot actual vs predicted
            plot_path = os.path.join(base_dir, "plots", f"{name}_actual_vs_pred.png")
            plt.figure(figsize=(6, 4))
            plt.scatter(y_test, y_pred, alpha=0.3)
            plt.xlabel("Actual Traffic Volume")
            plt.ylabel("Predicted Traffic Volume")
            plt.title(f"{name}: Actual vs Predicted")
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            # Log model
            mlflow.sklearn.log_model(model, artifact_path=f"{name}_model")

            print(f"[INFO] {name} -> RMSE: {rmse:.2f}, R2: {r2:.4f}")
