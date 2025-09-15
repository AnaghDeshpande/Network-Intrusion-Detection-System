from flask import Flask, render_template, request, jsonify
from src.index import run_model
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

app = Flask(__name__)

def sum_over_time(x):
    return tf.reduce_sum(x, axis=1)

def sum_over_time_output_shape(input_shape):
    return (input_shape[0], input_shape[2])

model = tf.keras.models.load_model(
    "nids_cnn_bilstm.h5",
    custom_objects={"sum_over_time": sum_over_time,
                    "sum_over_time_output_shape": sum_over_time_output_shape}
)

preproc = joblib.load("preprocessor.joblib")
label_enc = joblib.load("label_encoder.joblib")

# Example NSL-KDD row (same as dataset row but in DataFrame)
# row = "0,icmp,ecr_i,SF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,126,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,126,0.49,0.00,0.00,0.00,0.00,0.00,0.00,0.00,ipsweep,21"
# row = "0,tcp,ftp,SF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,2,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,portsweep,21"
# row = "0,tcp,auth,SF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,1,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,nmap,21"

# row = "0,udp,private,SF,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,2,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,teardrop,19"
# row = "0,icmp,ecr_i,SF,1032,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,511,511,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,255,1.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00,smurf,19"
# row = "0,tcp,ftp_data,SF,491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,25,0.00,0.00,0.00,0.00,0.20,0.05,0.00,255,25,0.10,0.05,0.00,0.00,0.00,0.00,0.00,0.00,neptune,19"


# row_list = row.split(",")

# Load column names (same as assign_column_names_if_possible)
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

# df = pd.DataFrame([row_list], columns=cols)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results')
def run():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = {}
    
    for col in cols:
        data[col] = 0  # Default value
    
    # Get form data
    for feature in request.form:
        if request.form[feature] != "":
            
            data[feature] = request.form[feature]
            
    df = pd.DataFrame([data])
    X = preproc.transform(df)

    # Predict
    probs = model.predict(X)
    # probs = pd.DataFrame([[0.02, 0.10, 0.70, 0.15, 0.03]])
    pred_idx = np.argmax(probs, axis=1)[0]
    pred_label = label_enc.inverse_transform([pred_idx])[0]

    print("Predicted:", pred_label)
    print("Probabilities:", probs[0])
    
    pairs = list(zip(label_enc.classes_, probs[0].tolist()))

    # return render_template(
    # "predict.html",
    # prediction=int(pred_idx),
    # label=pred_label,
    # pairs=pairs
    # )
    return render_template("predict.html", probs = probs, prediction=pred_label)
    # return render_template("predict.html", prediction = int(pred_idx), label=pred_label, probabilities=probs[0].tolist()) 

if __name__ == "__main__":
    app.run(debug=True)