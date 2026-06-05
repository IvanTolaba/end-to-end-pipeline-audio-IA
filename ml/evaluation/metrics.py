# ml/metrics.py
# ml/training/metrics.py
import json
import logging
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from config.settings import REPORT_TXT_PATH, REPORT_JSON_PATH, CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

def evaluate_and_save_numerical_metrics(y_true_onehot, y_pred_probs, y_true_classes, y_pred_classes) -> None:
    """Calculates model classification metrics safely, handling potential missing classes in tiny splits."""
    logger.info("Computing mathematical score parameters from inference predictions")
    
    # Calculate macro statistics across categories
    acc = accuracy_score(y_true_classes, y_pred_classes)
    prec = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    rec = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_true_onehot, y_pred_probs, average='macro', multi_class='ovr')
    except ValueError as e:
        logger.warning("ROC AUC calculation skipped or partial due to missing classes: %s", str(e))
        auc_score = float('nan')

    print("\n" + "="*50)
    print("🚀 PRODUCTION METRICS TEST RESULTS")
    print("="*50)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f} (Macro)")
    print(f"Recall    : {rec:.4f} (Macro)")
    print(f"F1-Score  : {f1:.4f} (Macro)")
    print(f"ROC AUC   : {auc_score}")
    print("="*50 + "\n")

    logger.info("Writing tabular verification matrix to file: %s", REPORT_TXT_PATH)
    raw_text_report = classification_report(
        y_true_classes, 
        y_pred_classes, 
        labels=list(range(NUM_CLASSES)), 
        target_names=CLASS_NAMES,
        zero_division=0
    )
    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as text_file:
        text_file.write(raw_text_report)

    logger.info("Writing structured verification json dictionary to file: %s", REPORT_JSON_PATH)
    metrics_payload = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_score_macro": float(f1),
        "roc_auc_ovr_macro": None if np.isnan(auc_score) else float(auc_score)
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(metrics_payload, json_file, indent=4)




