"""Weights & Biases integration for real-time inference monitoring."""
import wandb
import os
import logging
import time

logger = logging.getLogger(__name__)

_run = None


def init_wandb():
    """Initialize a W&B run for the backend inference server."""
    global _run
    api_key = os.getenv("WANDB_API_KEY", "")
    if not api_key:
        logger.warning("WANDB_API_KEY not set. W&B logging disabled.")
        return
    try:
        _run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "isl-detection"),
            name=f"backend-{os.getenv('HOSTNAME', 'local')}-{int(time.time())}",
            config={
                "model": "dynamic.tflite",
                "framework": "tflite",
                "environment": os.getenv("FLASK_ENV", "development"),
                "canary": os.getenv("CANARY", "false"),
            },
            tags=["inference", "production"],
            reinit=True,
        )
        logger.info("W&B initialized successfully.")
    except Exception as e:
        logger.error(f"W&B init failed: {e}")


def log_inference(gesture, confidence, latency_ms, endpoint):
    """Log a single inference event to W&B."""
    if _run is None:
        return
    try:
        wandb.log({
            "inference/gesture": gesture,
            "inference/confidence": confidence,
            "inference/latency_ms": latency_ms,
            "inference/endpoint": endpoint,
            "inference/timestamp": time.time(),
        })
    except Exception as e:
        logger.debug(f"W&B log failed: {e}")


def log_error(error_type, message):
    """Log errors to W&B for alerting."""
    if _run is None:
        return
    try:
        wandb.log({
            "errors/type": error_type,
            "errors/message": message,
            "errors/timestamp": time.time(),
        })
    except Exception:
        pass


def finish():
    """Gracefully close the W&B run."""
    global _run
    if _run:
        _run.finish()
        _run = None
