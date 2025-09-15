# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from data_processor import load_nsl_kdd, assign_column_names_if_possible, map_attack_category, preprocess_dataframe
# from model import sum_over_time, sum_over_time_output_shape
# from sklearn.metrics import confusion_matrix
# from tensorflow.keras.models import load_model

# # Load preprocessor and label encoder
# preproc = joblib.load("preprocessor.joblib")
# label_enc = joblib.load("label_encoder.joblib")

# # Load model (with custom objects if needed)
# model = load_model(
#     "nids_cnn_bilstm.h5",
#     custom_objects={"sum_over_time": sum_over_time,
#                     "sum_over_time_output_shape": sum_over_time_output_shape}
# )

# # Reload your test dataset
# df_test = load_nsl_kdd("data/NSL-KDD/KDDTest+.txt")
# df_test = assign_column_names_if_possible(df_test)
# df_test = map_attack_category(df_test, label_col="label")

# X_test, y_test, _, _, _ = preprocess_dataframe(df_test, categorical_cols=None, use_onehot=True)

# # Predict
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=label_enc.classes_,
#             yticklabels=label_enc.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()

from data_processor import load_nsl_kdd, assign_column_names_if_possible, map_attack_category, preprocess_dataframe

path = "data/NSL-KDD/KDDTrain+.txt"
df = load_nsl_kdd(path)
df = assign_column_names_if_possible(df)
df = map_attack_category(df, label_col="label")
print(df.head())
