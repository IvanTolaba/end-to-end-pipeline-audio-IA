import logging
from typing import List
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from config.settings import EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LEARNING_RATE

logger = logging.getLogger(__name__)

def get_training_callbacks() -> List[Callback]:
    """
    Provides early termination and learning rate attenuation strategies hooks.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1)
    return [early_stop, reduce_lr]