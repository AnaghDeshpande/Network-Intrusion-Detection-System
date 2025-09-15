import joblib
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
# from data_processor import load_nsl_kdd, assign_column_names_if_possible, map_attack_category, preprocess_dataframe
# from model import build_cnn_bilstm_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import confusion_matrix
    

from src.data_processor import load_nsl_kdd, assign_column_names_if_possible, map_attack_category, preprocess_dataframe
from src.model import build_cnn_bilstm_model

def train_pipeline(train_path, epochs=50, batch_size=128, seq_len=10, model_save_path="nids_cnn_bilstm.h5"):

    df = load_nsl_kdd(train_path)
    df = assign_column_names_if_possible(df)
    df = map_attack_category(df, label_col="label")

    X, y, preproc, label_enc, feature_name = preprocess_dataframe(df, categorical_cols=None, use_onehot=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    classes = np.unique(y_train)
    class_weights_values = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c):float(w) for c,w in zip(classes, class_weights_values)}

    input_dim = X_train.shape[1]
    num_classes = len(label_enc.classes_)
    model = build_cnn_bilstm_model(input_dim=input_dim, seq_len=seq_len, num_classes=num_classes)
    model.summary()

    opt = optimizers.Adam(learning_rate=1e-3)
    # f1_metric = tf.keras.metrics.F1Score(average="macro")
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    ]
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    # cb.append(lr_scheduler)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, class_weight=class_weight, callbacks=cb, verbose=2)
    
    results = model.evaluate(X_val, y_val, verbose=1, return_dict=True)
    
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)
    classification_Report = classification_report(y_val, y_pred, target_names=list(label_enc.classes_))
    print("classification_Report:\n\n", classification_Report)
    
    # Confusion Matrix
    print("\nConfusion Matrix: \n\n")
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    joblib.dump(preproc, "preprocessor.joblib")
    joblib.dump(label_enc, "label_encoder.joblib")

    print("Training complete. Model saved to", model_save_path)

    return model, preproc, label_enc, history, results


"""
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_enc.classes_, yticklabels=label_enc.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
"""