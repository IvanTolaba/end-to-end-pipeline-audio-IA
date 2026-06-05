import logging
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, LSTM, Dense, Bidirectional, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l2

from config.settings import (
    N_MFCC, NUM_TIMESTEPS, NUM_CLASSES, CONV_REGULARIZER_L2, 
    DENSE_REGULARIZER_L2, CNN_DROPOUT, LSTM_DROPOUT, FINAL_DROPOUT, 
    INITIAL_LEARNING_RATE
)

logger = logging.getLogger(__name__)

def build_cnn_blst() -> Sequential:
    """
    Assembles and compiles the model topology structural blocks.
    """
    logger.info("Assembling custom hybrid CNN-BiLSTM structural blocks")
    input_shape = (N_MFCC, NUM_TIMESTEPS, 1)
    model = Sequential()

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', 
                     kernel_regularizer=l2(CONV_REGULARIZER_L2), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(CNN_DROPOUT))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(CONV_REGULARIZER_L2)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(CNN_DROPOUT))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(CONV_REGULARIZER_L2)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(CNN_DROPOUT))

    model.add(Reshape((model.output_shape[1] * model.output_shape[2], model.output_shape[3])))

    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=LSTM_DROPOUT)))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=LSTM_DROPOUT)))
    model.add(GlobalAveragePooling1D())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(DENSE_REGULARIZER_L2)))
    model.add(Dropout(FINAL_DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    logger.info("Model compiled successfully")
    
    return model