import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

def attention_weighted_sum(rnn_seq, att_units=64):
    # rnn_seq shape: (batch, timesteps, features)
    score = layers.Dense(att_units, activation="tanh", kernel_regularizer=regularizers.l2(1e-4))(rnn_seq)
    score = layers.Dense(1, activation=None, kernel_regularizer=regularizers.l2(1e-4))(score)  # (batch, timesteps, 1)
    weights = layers.Softmax(axis=1, name="attention_weights")(score)  # sum over timesteps = 1
    weighted = layers.Multiply()([rnn_seq, weights])  # (batch, timesteps, features)
    out = tf.reduce_sum(weighted, axis=1)  # (batch, features)
    return out

def build_improved_model(input_dim,
                         seq_len=8,
                         num_classes=5,
                         embed_dim=128,
                         lstm_units=(96, 48),
                         dropout=0.45,
                         l2=1e-4,
                         small=False):
    """
    Improved CNN/BiLSTM-ish model for tabular NSL-KDD inputs (1D feature vectors).
    - Use Dense embedding, then RepeatVector to create a short pseudo-sequence for BiLSTM.
    - Stronger regularization (L2, dropout, recurrent_dropout).
    - Attention for pooling.
    """
    if small:
        embed_dim = max(64, embed_dim // 2)
        lstm_units = tuple(max(32, u//2) for u in lstm_units)
        dropout = min(0.5, dropout + 0.05)

    inp = layers.Input(shape=(input_dim,), name="input_features")

    # Input-level regularization
    x = layers.GaussianNoise(0.1, name="gauss_noise")(inp)
    x = layers.BatchNormalization()(x)

    # Dense embedding / bottleneck
    x = layers.Dense(embed_dim, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name="embed_dense")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # Repeat to create a short sequence for BiLSTM
    seq = layers.RepeatVector(seq_len, name="repeat_for_lstm")(x)

    # BiLSTM stack with recurrent dropout and L2
    y = seq
    for i, u in enumerate(lstm_units):
        y = layers.Bidirectional(
            layers.LSTM(u, return_sequences=True,
                        dropout=dropout*0.2,               # input dropout inside LSTM
                        recurrent_dropout=min(0.25, dropout*0.5),
                        kernel_regularizer=regularizers.l2(l2),
                        recurrent_regularizer=regularizers.l2(l2)),
            name=f"bilstm_{i+1}"
        )(y)
        y = layers.LayerNormalization()(y)  # stabilizes training

    # Attention pooling
    attended = attention_weighted_sum(y, att_units=64)
    attended = layers.BatchNormalization()(attended)

    # Final classifier head (smaller)
    h = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(attended)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2))(h)
    h = layers.Dropout(dropout*0.5)(h)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(h)

    model = models.Model(inputs=inp, outputs=outputs, name="Improved_CNN_BiLSTM_Attention")
    return model






# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models

# def sum_over_time(x):
#     return tf.reduce_sum(x, axis=1)

# def sum_over_time_output_shape(input_shape):
#     return (input_shape[0], input_shape[2])

# def build_cnn_bilstm_model(input_dim, seq_len=10, num_classes=5, cnn_filters=[64, 128], cnn_kernel=3,
#                             lstm_units=[128,64], dropout=0.4):
    
#     print("\nBuilding model...\n")
#     inp = layers.Input(shape=(input_dim,), name="input_features")
#     x = layers.Reshape((1, input_dim), name="expand")(inp)

#     # -------------------------------------------------------- CNN Feature Extraction --------------------------------------------------------
#     x = layers.Conv1D(filters=cnn_filters[0], kernel_size=cnn_kernel, padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv1D(filters=cnn_filters[-1], kernel_size=1, padding='same', activation='relu')(x)
#     x = layers.BatchNormalization()(x)

#     pooled_max = layers.GlobalMaxPooling1D()(x)
#     pooled_avg = layers.GlobalAveragePooling1D()(x)
#     cnn_feat = layers.Concatenate()([pooled_max, pooled_avg])
#     cnn_feat = layers.Dense(256, activation='relu')(cnn_feat)
#     cnn_feat = layers.Dropout(dropout)(cnn_feat)

#     # -------------------------------------------------------- Repeat for LSTM & Attention --------------------------------------------------------
#     seq = layers.RepeatVector(seq_len, name="repeat_for_lstm")(cnn_feat)

#     # -------------------------------------------------------- BiLSTM --------------------------------------------------------
#     y = seq
#     for i, u in enumerate(lstm_units):
#         y = layers.Bidirectional(
#             layers.LSTM(u, return_sequences=True, dropout=dropout),
#             name=f'bilstm_{i+1}'
#         )(y)
    
#     # -------------------------------------------------------- Attention --------------------------------------------------------
#     score = layers.Dense(64, activation="tanh")(y)
#     score = layers.Dense(1, activation="linear")(score)
#     attention_weights = layers.Softmax(axis=1, name="attention_weights")(score)
#     attended = layers.Multiply()([y, attention_weights])

#     attended = layers.Lambda(sum_over_time, output_shape=sum_over_time_output_shape, name="sum_over_time")(attended)

#     h = layers.Dense(128, activation='relu')(attended)
#     h = layers.Dropout(dropout)(h)
#     outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(h)

#     model = models.Model(inputs=inp, outputs=outputs, name="CNN_BiLSTM_Attention")

#     print("Model built successfully.\n")
#     return model

