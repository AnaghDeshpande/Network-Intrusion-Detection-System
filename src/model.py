import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def sum_over_time(x):
    return tf.reduce_sum(x, axis=1)

def sum_over_time_output_shape(input_shape):
    return (input_shape[0], input_shape[2])

def build_cnn_bilstm_model(input_dim, seq_len=10, num_classes=5, cnn_filters=[64, 128], cnn_kernel=3,
                            lstm_units=[128,64], dropout=0.4):
    
    print("\nBuilding model...\n")
    inp = layers.Input(shape=(input_dim,), name="input_features")
    x = layers.Reshape((1, input_dim), name="expand")(inp)

    # -------------------------------------------------------- CNN Feature Extraction --------------------------------------------------------
    x = layers.Conv1D(filters=cnn_filters[0], kernel_size=cnn_kernel, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=cnn_filters[-1], kernel_size=1, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    pooled_max = layers.GlobalMaxPooling1D()(x)
    pooled_avg = layers.GlobalAveragePooling1D()(x)
    cnn_feat = layers.Concatenate()([pooled_max, pooled_avg])
    cnn_feat = layers.Dense(256, activation='relu')(cnn_feat)
    cnn_feat = layers.Dropout(dropout)(cnn_feat)

    # -------------------------------------------------------- Repeat for LSTM & Attention --------------------------------------------------------
    seq = layers.RepeatVector(seq_len, name="repeat_for_lstm")(cnn_feat)

    # -------------------------------------------------------- BiLSTM --------------------------------------------------------
    y = seq
    for i, u in enumerate(lstm_units):
        y = layers.Bidirectional(
            layers.LSTM(u, return_sequences=True, dropout=dropout),
            name=f'bilstm_{i+1}'
        )(y)
    
    # -------------------------------------------------------- Attention --------------------------------------------------------
    score = layers.Dense(64, activation="tanh")(y)
    score = layers.Dense(1, activation="linear")(score)
    attention_weights = layers.Softmax(axis=1, name="attention_weights")(score)
    attended = layers.Multiply()([y, attention_weights])

    attended = layers.Lambda(sum_over_time, output_shape=sum_over_time_output_shape, name="sum_over_time")(attended)

    h = layers.Dense(128, activation='relu')(attended)
    h = layers.Dropout(dropout)(h)
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(h)

    model = models.Model(inputs=inp, outputs=outputs, name="CNN_BiLSTM_Attention")

    print("Model built successfully.\n")
    return model

