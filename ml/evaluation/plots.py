# ml/training/plots.py
# ml/training/plots.py
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg') # Enforce non-interactive backend context
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from config.settings import (
    PLOT_HISTORY_PATH, PLOT_MATRIX_PATH, PLOT_ROC_PATH, 
    HISTORY_LOSS_PATH, HISTORY_VAL_LOSS_PATH, NUM_CLASSES, CLASS_NAMES
)

plt.rcParams['font.family'] = 'DejaVu Sans'
logger = logging.getLogger(__name__)

def generate_and_save_plots(y_true_onehot, y_pred_probs, y_true_classes, y_pred_classes) -> None:
    """Execution router to compile and save verification data visualizations onto local storage."""
    plot_convergence_curves()
    plot_confusion_matrix(y_true_classes, y_pred_classes)
    plot_roc_curves(y_true_onehot, y_pred_probs)

def plot_convergence_curves() -> None:
    """Extracts accuracy and tracking parameters logs and exports convergence line plots."""
    if HISTORY_LOSS_PATH.exists() and HISTORY_VAL_LOSS_PATH.exists():
        logger.info("Rendering loss history charts curves to file: %s", PLOT_HISTORY_PATH)
        loss = np.load(HISTORY_LOSS_PATH)
        val_loss = np.load(HISTORY_VAL_LOSS_PATH)
        
        plt.figure(figsize=(5, 3.5))
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, '*', label='Training Loss')
        plt.plot(epochs, val_loss, '--', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Acoustic Model Convergence Curves')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOT_HISTORY_PATH, dpi=200)
        plt.close()

def plot_confusion_matrix(y_true, y_pred) -> None:
    """Builds and serializes a clean annotated confusion matrix graph mapping safely."""
    logger.info("Rendering model matrix confusion chart layout map to file: %s", PLOT_MATRIX_PATH)
    
    # FIX: Force confusion matrix to always be NUM_CLASSES x NUM_CLASSES (4x4) using explicit labels
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="left")
    ax.set_yticklabels(CLASS_NAMES)
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', 
                    color='black' if cm[i, j] < cm.max()/2 else 'white')
            
    plt.xlabel('Predicted Label Class')
    plt.ylabel('True Target Class')
    plt.title('Acoustic Model Confusion Matrix Map', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_MATRIX_PATH, dpi=200)
    plt.close()

def plot_roc_curves(y_true_onehot, y_pred_probs) -> None:
    """Computes Multi-class One-vs-Rest validation evaluation metrics safely handling missing classes."""
    logger.info("Rendering verification multi-class OVR ROC curve vector maps to file: %s", PLOT_ROC_PATH)
    plt.figure(figsize=(5.5, 4))

    for i in range(NUM_CLASSES):
        # FIX: Ensure y_true_onehot column has valid samples and contains more than 1 class state
        if i < y_true_onehot.shape[1] and len(np.unique(y_true_onehot[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
        else:
            logger.warning("Class %s not present in evaluation ground truth slice. Skipping line plotting.", CLASS_NAMES[i])
            plt.plot([], [], label=f'{CLASS_NAMES[i]} (AUC = N/A)')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Multi-Class ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_ROC_PATH, dpi=200)
    plt.close()

