# ml/training/evaluate.py
import logging
import numpy as np
from keras.models import load_model

from config.settings import MODEL_SAVE_PATH, X_TEST_CACHE_PATH, y_TEST_CACHE_PATH
# 🔹 FIX: Target exact relative package paths inside ml/training layout
from ml.evaluation.metrics import evaluate_and_save_numerical_metrics
from ml.evaluation.plots import generate_and_save_plots

logger = logging.getLogger(__name__)

def run_standalone_evaluation() -> None:
    """
    Orchestrates headless metric derivations and assets 
    generation using pre-saved tensors.
    """
    logger.info("Initializing standalone isolated performance valuation context workflow")
    
    # Verify previous dependency compliance requirements are met
    if not MODEL_SAVE_PATH.exists() or not X_TEST_CACHE_PATH.exists():
        logger.error("Missing execution structures. Run python -m ml.training.train first.")
        raise FileNotFoundError("Prerequisite static binaries artifacts were not found on disk.")

    # Ingest baseline structural components
    logger.info("Ingesting model architecture and cache numpy matrices from serialization layer")
    model = load_model(MODEL_SAVE_PATH)
    X_test = np.load(X_TEST_CACHE_PATH)
    y_test_onehot = np.load(y_TEST_CACHE_PATH)

    # Execute predictions across evaluation vectors
    logger.info("Running network forward-pass probability estimations loop")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_test_classes = np.argmax(y_test_onehot, axis=1)

    # 1. Trigger numeric extraction and file storage routines
    evaluate_and_save_numerical_metrics(y_test_onehot, y_pred_probs, y_test_classes, y_pred_classes)

    # 2. Trigger automated graph visualization engines
    generate_and_save_plots(y_test_onehot, y_pred_probs, y_test_classes, y_pred_classes)
    
    logger.info("🎉 Separate analytical validation step execution context successfully completed.")

if __name__ == "__main__":
    run_standalone_evaluation()


