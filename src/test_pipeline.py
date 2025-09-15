from tensorflow.keras.models import load_model
import joblib
import numpy as np
from data_processor import load_nsl_kdd, assign_column_names_if_possible, map_attack_category, preprocess_dataframe
from model import sum_over_time, sum_over_time_output_shape
from sklearn.metrics import classification_report

def test_pipeline(test_path, model_path="nids_cnn_bilstm.h5", preproc_path="preprocessor.joblib", label_enc_path="label_encoder.joblib"):
    # Load test data
    df_test = load_nsl_kdd(test_path)
    df_test = assign_column_names_if_possible(df_test)
    df_test = map_attack_category(df_test, label_col="label")

    # Load preprocessing tools
    preproc = joblib.load(preproc_path)
    label_enc = joblib.load(label_enc_path)

    # Apply same preprocessing (do NOT refit!)
    X_test, y_test, _, _, _ = preprocess_dataframe(df_test, categorical_cols=None, use_onehot=True)
    
    # Make sure preprocessing is consistent
    X_test = preproc.transform(df_test.drop(columns=["attack_category", "label"]))
    y_test = label_enc.transform(df_test["attack_category"])

    # Load trained model
    model = load_model(
        "nids_cnn_bilstm.h5",
        custom_objects={
            "sum_over_time": sum_over_time,
            "sum_over_time_output_shape": sum_over_time_output_shape
        }
    )

    # Evaluate
    results = model.evaluate(X_test, y_test, verbose=1, return_dict=True)

    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=list(label_enc.classes_))
    print("Test Results:", results)
    print("Classification Report:\n", report)

    return results, report

if __name__ == "__main__":
    test_path = "data/NSL-KDD/KDDTest+.txt"
    test_pipeline(test_path)