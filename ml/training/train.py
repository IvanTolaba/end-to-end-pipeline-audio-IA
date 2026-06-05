import logging
import numpy as np

from config.settings import DATA_PROCESSED_DIR, TRAINING_EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH, HISTORY_LOSS_PATH, HISTORY_VAL_LOSS_PATH
from ml.training.preprocessing import load_parquet_layer, balance_and_split_pipeline
from ml.training.architectures import build_cnn_blst
from ml.training.callbacks import get_training_callbacks

logger = logging.getLogger(__name__)

def execute_model_training() -> None:
    """
    Orchestrates the end-to-end model training pipeline.
    Loads processed features, balances datasets, trains the neural network 
    and serializes convergence logs and weights.
    """
    # 1. Ingestion 
    logger.info("Ingesting processed Silver layer datasets...")
    df_silver = load_parquet_layer(DATA_PROCESSED_DIR)
    
    # 2. Prepare datasets
    X_train, y_train, X_val, y_val, _, _ = balance_and_split_pipeline(df_silver)

    # 3. Build model
    logger.info("Building model...")
    nn_model = build_cnn_blst()
    callbacks = get_training_callbacks()

    # 4. Train
    logger.info("Starting training (epochs=%s)", TRAINING_EPOCHS)
    history = nn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TRAINING_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Save artifacts
    logger.info("Saving training history...")
    np.save(HISTORY_LOSS_PATH, np.array(history.history['loss']))
    np.save(HISTORY_VAL_LOSS_PATH, np.array(history.history['val_loss']))

    logger.info("Saving optimized .keras model to: %s", MODEL_SAVE_PATH)
    nn_model.save(MODEL_SAVE_PATH)
    logger.info("🎉 Model training pipeline completed successfully.")

if __name__ == "__main__":
    execute_model_training()