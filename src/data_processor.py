import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --------- helpers: load + map labels ----------

def load_nsl_kdd(file_path):
    print("\nLoading data from:\t", file_path)
    df = pd.read_csv(file_path, header = None, engine='python')
    if df.shape[1] == 1:
        df = pd.read_csv(file_path, header=None, delim_whitespace=True, engine='python')
    
    print("Data shape:\t", df.shape)
    return df

def assign_column_names_if_possible(df):
    ncols = df.shape[1]
    if ncols == 43:
        names = [
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
            "label",              # attack name (normal, neptune, smurf, etc.)
            "difficulty_level"    # integer 0–20
        ]   
    else:
        names = [f"feature_{i}" for i in range(ncols-1)] + ["label"]
    df.columns = names
    print(df.columns)
    return df

def map_attack_category(df, label_col="label"):
    attack_map = {
        # dos
        'neptune':'dos','back':'dos','land':'dos','pod':'dos','smurf':'dos','teardrop':'dos',
        'mailbomb':'dos','apache2':'dos','processtable':'dos','udpstorm':'dos',
        # probe
        'satan':'probe','ipsweep':'probe','nmap':'probe','portsweep':'probe','mscan':'probe','saint':'probe',
        # r2l
        'guess_passwd':'r2l','ftp_write':'r2l','imap':'r2l','phf':'r2l','multihop':'r2l',
        'warezmaster':'r2l','warezclient':'r2l','spy':'r2l','snmpguess':'r2l','snmpgetattack':'r2l',
        'httptunnel':'r2l','sendmail':'r2l','named':'r2l',
        # u2r
        'buffer_overflow':'u2r','loadmodule':'u2r','rootkit':'u2r','perl':'u2r',
        'sqlattack':'u2r','xterm':'u2r','ps':'u2r'
    }
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame columns.")
    
    df['attack_type'] = df[label_col].astype(str).str.strip().str.lower()
    df["attack_category"] = df["attack_type"].map(attack_map).fillna("normal")
    return df

# --------- preprocessing ---------- 

def preprocess_dataframe(df, categorical_cols=None, use_onehot=True):
    print("\nPreprocessing data...\n\n")

    if "attack_category" not in df.columns:
        raise ValueError("DataFrame must contain 'attack_category' column. Please run map_attack_category first.")
    if categorical_cols is None:
        possible = []
        for name in ["protocol_type", "service", "flag"]:
            if name in df.columns:
                possible.append(name)
        if not possible:
            cols = list(df.columns)
            if len(cols) >= 4 :
                possible = [cols[1], cols[2], cols[3]]
        categorical_cols = possible
    
    exclude = set(categorical_cols + ["label", "attack_type", "attack_category", "difficulty_level"])
    numeric_cols = [col for col in df.columns if col not in exclude]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    
    if use_onehot:
        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore')
        transformers = [("num", StandardScaler(), numeric_cols),
                        ("cat", ohe, categorical_cols)]
    else:
        transformers = [("num", StandardScaler(), numeric_cols)]
    col_transformer = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)

    X = col_transformer.fit_transform(df.drop(columns=["attack_type", "attack_category"], errors='ignore'))
    feature_names = []
    feature_names.extend(numeric_cols)
    if use_onehot:
        try:
            cat_names = col_transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        except Exception:
            cat_names = []
        feature_names.extend(list(cat_names))
    
    le = LabelEncoder()
    y = le.fit_transform(df["attack_category"].astype(str))

    # print("mapping", dict(zip(le.classes_, le.transform(le.classes_))))
    # print("Feature names:\t", feature_names)
    # print("\nClasses:\t", le.classes_)  
    # # print("\nX shape:\t", X[5], "\ty shape:\t", y[5])
    # print("col_transformer\t", col_transformer)
    print("\nPreprocessing complete.\n")
    return X, y, col_transformer, le, feature_names