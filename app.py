from flask import Flask, render_template, request, jsonify, send_file
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- Custom objects used for the model ---
def sum_over_time(x):
    return tf.reduce_sum(x, axis=1)

def sum_over_time_output_shape(input_shape):
    return (input_shape[0], input_shape[2])

# --- Load model and preprocessors ---
model = tf.keras.models.load_model(
    "nids_cnn_bilstm.h5",
    custom_objects={"sum_over_time": sum_over_time,
                    "sum_over_time_output_shape": sum_over_time_output_shape}
)

preproc = joblib.load("preprocessor.joblib")
label_enc = joblib.load("label_encoder.joblib")

# Column names
cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label","difficulty_level"
]

# Attack category mapping based on NSL-KDD dataset
attack_category_mapping = {
    'normal': 'Normal',
    # DoS attacks
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'apache2': 'DoS',
    'udpstorm': 'DoS',
    'processtable': 'DoS',
    'mailbomb': 'DoS',
    # Probe attacks
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',
    # R2L (Remote to Local) attacks
    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',
    # U2R (User to Root) attacks
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}

def get_attack_category(attack_label):
    """Convert specific attack to broader category"""
    attack_lower = str(attack_label).lower()
    return attack_category_mapping.get(attack_lower, 'Unknown')

def calculate_category_confidences(probs):
    """
    Calculate confidence scores for each of the 5 categories by summing probabilities
    Returns a dictionary with category confidences for each sample
    """
    num_samples = len(probs)
    
    # Initialize confidence arrays for each category
    category_confidences = {
        'Normal': np.zeros(num_samples),
        'DoS': np.zeros(num_samples),
        'Probe': np.zeros(num_samples),
        'R2L': np.zeros(num_samples),
        'U2R': np.zeros(num_samples)
    }
    
    # Sum probabilities by category (aggregate all attacks in same category)
    for i, class_name in enumerate(label_enc.classes_):
        category = get_attack_category(class_name)
        
        if category in category_confidences:
            category_confidences[category] += probs[:, i]
    
    return category_confidences

# -------------------------
# Helper: prepare df for preproc
# -------------------------
def prepare_df_for_preproc(df, preproc, expected_cols=None):
    """
    Make incoming df compatible with fitted ColumnTransformer 'preproc'.
    - Replace pd.NA with np.nan
    - Ensure expected_cols exist (create missing with NaN) and drop extras
    - Detect categorical columns used by OneHotEncoder in the preproc and coerce them to object dtype
    - Coerce numeric-like columns to numeric where possible
    Returns (cleaned_df, detected_cat_cols)
    """
    df = df.copy()

    # Replace pandas nullable NA with np.nan
    df.replace({pd.NA: np.nan}, inplace=True)

    # If expected column list provided, ensure those columns exist and keep only them
    if expected_cols is not None:
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan
        # keep only expected cols in given order (if they exist)
        df = df.loc[:, [c for c in expected_cols if c in df.columns]]

    # Attempt to detect categorical columns from preproc.transformers_
    cat_cols = []
    try:
        for name, transformer, cols_in_transformer in preproc.transformers_:
            if cols_in_transformer == 'drop' or cols_in_transformer == 'passthrough':
                continue

            found_ohe = False
            # direct OHE
            if isinstance(transformer, OneHotEncoder) or hasattr(transformer, 'categories_'):
                found_ohe = True
            else:
                # pipeline-like: check named_steps
                if hasattr(transformer, 'named_steps'):
                    for step in transformer.named_steps.values():
                        if isinstance(step, OneHotEncoder) or hasattr(step, 'categories_'):
                            found_ohe = True
                            break

            if found_ohe:
                # cols_in_transformer may be list of strings or indices
                if isinstance(cols_in_transformer, (list, tuple, np.ndarray)):
                    for c in cols_in_transformer:
                        if isinstance(c, (int, np.integer)):
                            # try to map index to column name if df has that many columns
                            try:
                                cat_cols.append(df.columns[int(c)])
                            except Exception:
                                pass
                        else:
                            cat_cols.append(c)
                else:
                    # single column name / index
                    if isinstance(cols_in_transformer, (int, np.integer)):
                        try:
                            cat_cols.append(df.columns[int(cols_in_transformer)])
                        except Exception:
                            pass
                    else:
                        cat_cols.append(cols_in_transformer)
    except Exception:
        cat_cols = []

    # Deduplicate cat_cols and keep only those present in df
    cat_cols = [c for i, c in enumerate(cat_cols) if c in df.columns and c not in cat_cols[:i]]

    # Make categorical columns object dtype and ensure missing as np.nan
    for c in cat_cols:
        try:
            df[c] = df[c].astype(object)
            df[c] = df[c].where(df[c].notnull(), np.nan)
        except Exception:
            # best-effort: convert with errors ignored
            df[c] = df[c].astype(object, errors='ignore')

    # Coerce non-categorical columns to numeric where sensible (leave categorical untouched)
    for c in df.columns:
        if c not in cat_cols:
            # Attempt conversion; don't coerce strings like 'tcp' etc.
            df[c] = pd.to_numeric(df[c], errors='ignore')

    return df, cat_cols

# -------------------------
# Routes
# -------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results')
def run():
    return render_template("predict.html")

@app.route('/category')
def category():
    return render_template("category.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Single-row prediction from form inputs
    data = {}
    for col in cols:
        data[col] = 0  # Default value

    # Get form data
    for feature in request.form:
        if request.form[feature] != "":
            data[feature] = request.form[feature]

    df = pd.DataFrame([data])
    # Prepare df for preproc
    df_clean, detected_cat_cols = prepare_df_for_preproc(df, preproc, expected_cols=cols)

    try:
        X = preproc.transform(df_clean)
    except Exception as e:
        # Provide diagnostic info
        print("Error during preproc.transform in /predict:", str(e))
        print("Dtypes:\n", df_clean.dtypes)
        for c in df_clean.columns:
            print(f"{c} sample:", df_clean[c].head(5).tolist())
        return jsonify({"error": f"Preprocessing error: {str(e)}"}), 500

    probs = model.predict(X)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    pred_label = label_enc.inverse_transform([pred_idx])[0]

    print("Predicted:", pred_label)
    print("Probabilities:", probs[0])

    confidence_score = list(zip(label_enc.classes_, probs[0].tolist()))
    confidence_score.sort(key=lambda x: x[1], reverse=True)

    return render_template("predict.html", prediction=pred_idx, label=pred_label, confidence_score=confidence_score)

@app.route("/predict_category", methods=["POST"])
def predict_category():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"category_input_{timestamp}.csv")
        file.save(upload_path)

        # Read CSV file
        df = pd.read_csv(upload_path)

        # If CSV has same number of columns as cols list but wrong header, rename
        if df.shape[1] == len(cols):
            if not all(df.columns == cols):
                df.columns = cols

        # Prepare df for the fitted preprocessor
        df_clean, detected_cat_cols = prepare_df_for_preproc(df, preproc, expected_cols=cols)

        # Debug prints (helpful if transform fails)
        print("Incoming CSV shape:", df.shape)
        print("Cleaned DF shape:", df_clean.shape)
        print("Detected categorical columns for preproc:", detected_cat_cols)
        print("Dtypes after cleaning:\n", df_clean.dtypes)

        # Transform
        try:
            X = preproc.transform(df_clean)
        except Exception as e:
            print("Error during preproc.transform:", str(e))
            print("Sample values (first 5 rows) for columns and dtypes:")
            for c in df_clean.columns:
                vals = df_clean[c].head(5).tolist()
                print(f"  {c}  dtype={df_clean[c].dtype}  sample={vals}")
            raise

        # Predict
        probs = model.predict(X)
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = label_enc.inverse_transform(pred_indices)

        # Get predicted category for each sample
        pred_categories = [get_attack_category(label) for label in pred_labels]

        # Get confidence scores for predicted specific attack
        specific_confidence_scores = np.max(probs, axis=1)

        # Calculate confidence scores for all 5 categories
        category_confidences = calculate_category_confidences(probs)

        # Create detailed confidence scores for each sample (similar to predict function)
        all_class_confidences = []
        for i in range(len(probs)):
            sample_confidences = list(zip(label_enc.classes_, probs[i].tolist()))
            sample_confidences.sort(key=lambda x: x[1], reverse=True)
            all_class_confidences.append(sample_confidences)

        # Create output DataFrame with all information
        output_df = pd.DataFrame({
            'specific_attack': pred_labels,
            # 'predicted_category': pred_categories,
            # 'specific_attack_confidence': specific_confidence_scores,
            # 'normal_confidence': category_confidences['Normal'],
            # 'dos_confidence': category_confidences['DoS'],
            # 'probe_confidence': category_confidences['Probe'],
            # 'r2l_confidence': category_confidences['R2L'],
            # 'u2r_confidence': category_confidences['U2R']
        })

        # Add top 5 specific attack predictions with confidences for each sample
        for rank in range(min(5, len(label_enc.classes_))):
            output_df[f'top_{rank+1}_attack'] = [conf[rank][0] if rank < len(conf) else '' 
                                                   for conf in all_class_confidences]
            output_df[f'top_{rank+1}_confidence'] = [conf[rank][1] if rank < len(conf) else 0.0 
                                                      for conf in all_class_confidences]

        # Save output CSV with full precision
        output_filename = f"category_predictions_{timestamp}.csv"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        output_df.to_csv(output_path, index=False, float_format='%.15f')

        print(f"Processed {len(df)} rows")
        print(f"Output saved to: {output_path}")
        print(f"Sample category confidences for first row:")
        print(f"  Normal: {category_confidences['Normal'][0]:.6f}")
        print(f"  DoS: {category_confidences['DoS'][0]:.6f}")
        print(f"  Probe: {category_confidences['Probe'][0]:.6f}")
        print(f"  R2L: {category_confidences['R2L'][0]:.6f}")
        print(f"  U2R: {category_confidences['U2R'][0]:.6f}")

        # Return the file for download
        return send_file(
            output_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=output_filename
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)





# from flask import Flask, render_template, request, jsonify, send_file
# import joblib
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['OUTPUT_FOLDER'] = 'outputs'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# # Create folders if they don't exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# def sum_over_time(x):
#     return tf.reduce_sum(x, axis=1)

# def sum_over_time_output_shape(input_shape):
#     return (input_shape[0], input_shape[2])

# model = tf.keras.models.load_model(
#     "nids_cnn_bilstm.h5",
#     custom_objects={"sum_over_time": sum_over_time,
#                     "sum_over_time_output_shape": sum_over_time_output_shape}
# )

# preproc = joblib.load("preprocessor.joblib")
# label_enc = joblib.load("label_encoder.joblib")

# # Column names
# cols = [
#     "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
#     "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
#     "root_shell","su_attempted","num_root","num_file_creations","num_shells",
#     "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
#     "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
#     "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
#     "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
#     "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
#     "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
#     "dst_host_srv_rerror_rate",
#     "label","difficulty_level"
# ]

# # Attack category mapping based on NSL-KDD dataset
# attack_category_mapping = {
#     'normal': 'Normal',
#     # DoS attacks
#     'back': 'DoS',
#     'land': 'DoS',
#     'neptune': 'DoS',
#     'pod': 'DoS',
#     'smurf': 'DoS',
#     'teardrop': 'DoS',
#     'apache2': 'DoS',
#     'udpstorm': 'DoS',
#     'processtable': 'DoS',
#     'mailbomb': 'DoS',
#     # Probe attacks
#     'ipsweep': 'Probe',
#     'nmap': 'Probe',
#     'portsweep': 'Probe',
#     'satan': 'Probe',
#     'mscan': 'Probe',
#     'saint': 'Probe',
#     # R2L (Remote to Local) attacks
#     'ftp_write': 'R2L',
#     'guess_passwd': 'R2L',
#     'imap': 'R2L',
#     'multihop': 'R2L',
#     'phf': 'R2L',
#     'spy': 'R2L',
#     'warezclient': 'R2L',
#     'warezmaster': 'R2L',
#     'sendmail': 'R2L',
#     'named': 'R2L',
#     'snmpgetattack': 'R2L',
#     'snmpguess': 'R2L',
#     'xlock': 'R2L',
#     'xsnoop': 'R2L',
#     'worm': 'R2L',
#     # U2R (User to Root) attacks
#     'buffer_overflow': 'U2R',
#     'loadmodule': 'U2R',
#     'perl': 'U2R',
#     'rootkit': 'U2R',
#     'httptunnel': 'U2R',
#     'ps': 'U2R',
#     'sqlattack': 'U2R',
#     'xterm': 'U2R'
# }

# def get_attack_category(attack_label):
#     """Convert specific attack to broader category"""
#     attack_lower = attack_label.lower()
#     return attack_category_mapping.get(attack_lower, 'Unknown')

# def calculate_category_confidences(probs):
#     """
#     Calculate confidence scores for each of the 5 categories by summing probabilities
#     Returns a dictionary with category confidences for each sample
#     """
#     num_samples = len(probs)
    
#     # Initialize confidence arrays for each category
#     category_confidences = {
#         'Normal': np.zeros(num_samples),
#         'DoS': np.zeros(num_samples),
#         'Probe': np.zeros(num_samples),
#         'R2L': np.zeros(num_samples),
#         'U2R': np.zeros(num_samples)
#     }
    
#     # Sum probabilities by category (aggregate all attacks in same category)
#     for i, class_name in enumerate(label_enc.classes_):
#         category = get_attack_category(class_name)
        
#         if category in category_confidences:
#             category_confidences[category] += probs[:, i]
    
#     return category_confidences

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/results')
# def run():
#     return render_template("predict.html")

# @app.route('/category')
# def category():
#     return render_template("category.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = {}
    
#     for col in cols:
#         data[col] = 0  # Default value
    
#     # Get form data
#     for feature in request.form:
#         if request.form[feature] != "":
#             data[feature] = request.form[feature]
            
#     df = pd.DataFrame([data])
#     X = preproc.transform(df)

#     # Predict
#     probs = model.predict(X)
#     pred_idx = np.argmax(probs, axis=1)[0]
#     pred_label = label_enc.inverse_transform([pred_idx])[0]

#     print("Predicted:", pred_label)
#     print("Probabilities:", probs[0])
    
#     confidence_score = list(zip(label_enc.classes_, probs[0].tolist()))
#     confidence_score.sort(key=lambda x: x[1], reverse=True)

#     return render_template("predict.html", prediction=int(pred_idx), label=pred_label, confidence_score=confidence_score)


# @app.route("/predict_category", methods=["POST"])
# def predict_category():
#     try:
#         # Check if file was uploaded
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
        
#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({"error": "No file selected"}), 400
        
#         if not file.filename.endswith('.csv'):
#             return jsonify({"error": "File must be a CSV"}), 400
        
#         # Save uploaded file
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"category_input_{timestamp}.csv")
#         file.save(upload_path)
        
#         # Read CSV file
#         df = pd.read_csv(upload_path)
        
#         # Check if CSV has column names, if not assign them
#         if df.shape[1] == len(cols):
#             if not all(df.columns == cols):
#                 df.columns = cols
        
#         # Preprocess the data
#         X = preproc.transform(df)
        
#         # Make predictions
#         probs = model.predict(X)
#         pred_indices = np.argmax(probs, axis=1)
#         pred_labels = label_enc.inverse_transform(pred_indices)
        
#         # Get predicted category for each sample
#         pred_categories = [get_attack_category(label) for label in pred_labels]
        
#         # Get confidence scores for predicted specific attack
#         specific_confidence_scores = np.max(probs, axis=1)
        
#         # Calculate confidence scores for all 5 categories
#         category_confidences = calculate_category_confidences(probs)
        
#         # Create detailed confidence scores for each sample (similar to predict function)
#         all_class_confidences = []
#         for i in range(len(probs)):
#             sample_confidences = list(zip(label_enc.classes_, probs[i].tolist()))
#             sample_confidences.sort(key=lambda x: x[1], reverse=True)
#             all_class_confidences.append(sample_confidences)
        
#         # Create output DataFrame with all information
#         output_df = pd.DataFrame({
#             'specific_attack': pred_labels,
#             'predicted_category': pred_categories,
#             'specific_attack_confidence': specific_confidence_scores,
#             'normal_confidence': category_confidences['Normal'],
#             'dos_confidence': category_confidences['DoS'],
#             'probe_confidence': category_confidences['Probe'],
#             'r2l_confidence': category_confidences['R2L'],
#             'u2r_confidence': category_confidences['U2R']
#         })
        
#         # Add top 5 specific attack predictions with confidences for each sample
#         for rank in range(min(5, len(label_enc.classes_))):
#             output_df[f'top_{rank+1}_attack'] = [conf[rank][0] if rank < len(conf) else '' 
#                                                    for conf in all_class_confidences]
#             output_df[f'top_{rank+1}_confidence'] = [conf[rank][1] if rank < len(conf) else 0.0 
#                                                       for conf in all_class_confidences]
        
#         # Save output CSV with full precision
#         output_filename = f"category_predictions_{timestamp}.csv"
#         output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
#         output_df.to_csv(output_path, index=False, float_format='%.15f')
        
#         print(f"Processed {len(df)} rows")
#         print(f"Output saved to: {output_path}")
#         print(f"Sample category confidences for first row:")
#         print(f"  Normal: {category_confidences['Normal'][0]:.6f}")
#         print(f"  DoS: {category_confidences['DoS'][0]:.6f}")
#         print(f"  Probe: {category_confidences['Probe'][0]:.6f}")
#         print(f"  R2L: {category_confidences['R2L'][0]:.6f}")
#         print(f"  U2R: {category_confidences['U2R'][0]:.6f}")
        
#         # Return the file for download
#         return send_file(
#             output_path,
#             mimetype='text/csv',
#             as_attachment=True,
#             download_name=output_filename
#         )
        
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)