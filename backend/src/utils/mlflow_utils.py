"""Utility to integrate with MLflow Model Registry for model versioning."""
import mlflow
import os
import logging

logger = logging.getLogger(__name__)


def init_mlflow():
    """Set the MLflow tracking URI from environment variable."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if uri:
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI set to: {uri}")
    else:
        logger.warning("MLFLOW_TRACKING_URI not set, using local models only.")


def log_prediction(gesture, confidence, latency_ms):
    """Log inference metrics to MLflow (best-effort, non-blocking)."""
    try:
        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.log_metrics({
                "inference_confidence": confidence,
                "inference_latency_ms": latency_ms,
            })
    except Exception as e:
        logger.debug(f"MLflow logging skipped: {e}")
