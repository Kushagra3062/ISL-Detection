"""Register existing .h5 and .tflite models into MLflow Model Registry."""
import mlflow
import os
import sys


MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_URI)


def register_model(model_path, model_name, run_name):
    """Log a model file as an MLflow artifact and register it."""
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        return

    mlflow.set_experiment("isl-detection-models")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(model_path)
        mlflow.log_param("model_file", os.path.basename(model_path))
        mlflow.log_param("model_size_mb", round(os.path.getsize(model_path) / 1e6, 2))

        # Register in Model Registry
        artifact_uri = mlflow.get_artifact_uri(os.path.basename(model_path))
        mlflow.register_model(artifact_uri, model_name)

    print(f"✅ Registered '{model_name}' from '{model_path}'")


if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    register_model(
        os.path.join(models_dir, "Dynamic.h5"),
        "isl-dynamic-keras",
        "register-dynamic-h5",
    )
    register_model(
        os.path.join(models_dir, "dynamic.tflite"),
        "isl-dynamic-tflite",
        "register-dynamic-tflite",
    )
    register_model(
        os.path.join(models_dir, "landmark_model.h5"),
        "isl-landmark-keras",
        "register-landmark-h5",
    )
