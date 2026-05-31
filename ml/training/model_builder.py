import logging
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, BatchNormalization, 
                                     Dropout, Reshape, LSTM, Dense, 
                                     Bidirectional, GlobalAveragePooling1D)
from keras.optimizers import Adam
from keras.regularizers import l2

logger = logging.getLogger(__name__)

def build_cnn_bilstm_model(input_shape=(13, 130, 1), num_classes=4) -> Sequential:
    """
    Construye y compila la arquitectura de red neuronal profunda híbrida CNN-BLSTM.
    """
    logger.info(f"Instanciando red neuronal con input_shape={input_shape}")
    model = Sequential()
    
    # 🔹 CNN Bloque 1
    model.add(Conv2D(128, (3,3), activation='relu', padding='same', 
                     kernel_regularizer=l2(0.005), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    # 🔹 CNN Bloque 2
    model.add(Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    # 🔹 CNN Bloque 3
    model.add(Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    # 🔹 Acondicionamiento para pasar de características espaciales a secuencias temporales
    model.add(Reshape((model.output_shape[1] * model.output_shape[2], model.output_shape[3])))
    
    # 🔹 Bloque Recurrente Avanzado (BiLSTM)
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)))
    
    # 🔹 Reducción global y capas de clasificación
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    logger.info("Modelo compilado exitosamente.")
    return model