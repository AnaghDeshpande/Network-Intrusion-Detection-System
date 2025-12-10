import os
from collections import Counter
import joblib

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE


# ---------------- utilities ----------------

def load_nsl_kdd(path, sep=None):
    """
    Robust loader: tries CSV then whitespace-delimited if necessary.
    """
    print(f"Loading: {path}")
    if sep is None:
        try:
            df = pd.read_csv(path, header=None, engine='python')
            # if everything ended up in one column, retry with delim_whitespace
            if df.shape[1] == 1:
                df = pd.read_csv(path, header=None, delim_whitespace=True, engine='python')
        except Exception:
            df = pd.read_csv(path, header=None, delim_whitespace=True, engine='python')
    else:
        df = pd.read_csv(path, header=None, sep=sep, engine='python')
    print("Loaded shape:", df.shape)
    return df


def assign_column_names_if_possible(df):
    """
    Assigns the 43 NSL-KDD feature names if dataframe has 43 or 42 columns (sometimes no difficulty column).
    Otherwise assigns generic names and makes sure last column is 'label'.
    """
    ncols = df.shape[1]
    if ncols in (43, 42):
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
            "label"
        ]
        # If file has 43 columns, there is a difficulty_level column after label in some formats
        if ncols == 43:
            names = names + ["difficulty_level"]
        df.columns = names
    else:
        # Last column is assumed target label
        names = [f"feature_{i}" for i in range(ncols-1)] + ["label"]
        df.columns = names

    print("Columns set. Example columns:", list(df.columns[:8]))
    return df


def map_attack_category(df, label_col="label", drop_original=False):
    """
    Map raw attack names to categories: 'dos','probe','r2l','u2r','normal'.
    Leaves unknowns as 'normal' (consistent with many NSL-KDD mappings).
    """
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
        raise ValueError(f"Label column '{label_col}' not found.")
    df['attack_type'] = df[label_col].astype(str).str.strip().str.lower()
    df['attack_category'] = df['attack_type'].map(attack_map).fillna('normal')
    if drop_original:
        df = df.drop(columns=[label_col])
    return df


def reduce_cardinality(series, top_k=30):
    """
    Keep top_k most frequent categories, rest -> '__OTHER__'.
    """
    top = set(series.value_counts().nlargest(top_k).index)
    return series.apply(lambda x: x if x in top else "__OTHER__")


# ---------------- preprocessing pipeline builder ----------------

def build_preprocessing_pipeline(
    df,
    categorical_cols=None,
    numeric_strategy='median',
    rare_thresh=30,
    ohe_drop='first',
    scaler=RobustScaler()
):
    """
    Build a sklearn Pipeline + ColumnTransformer and return it UNFITTED.
    - categorical_cols: list or None; if None we look for expected columns.
    - rare_thresh (int): keep top-K categories for each categorical col, rest -> '__OTHER__'
    - ohe_drop: passed to OneHotEncoder (can be None or 'first')
    """
    # auto detect categorical columns if not provided
    if categorical_cols is None:
        possible = []
        for name in ["protocol_type", "service", "flag"]:
            if name in df.columns:
                possible.append(name)
        categorical_cols = possible

    # exclude meta cols from numeric features
    exclude = set(categorical_cols + ["label", "attack_type", "attack_category", "difficulty_level"])
    numeric_cols = [c for c in df.columns if c not in exclude]

    # numeric subpipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler", scaler),
    ])

    # categorical subpipeline
    cat_pipeline = Pipeline([
        # OneHotEncoder will handle unknowns during transform
        ("ohe", OneHotEncoder(drop=ohe_drop, sparse_output=False, handle_unknown='ignore'))
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    return transformer, numeric_cols, categorical_cols


# ---------------- fit/transform, train/test splitting, SMOTE ----------------

def prepare_data_pipeline(
    df,
    categorical_cols=None,
    test_size=0.2,
    val_size=None,
    random_state=42,
    apply_smote=False,
    smote_k_neighbors=5,
    rare_topk=30,
    save_pipeline_path=None
):
    """
    Full prepare function:
    - maps types (converts numerics),
    - reduces cardinality on categorical columns (top-K -> keep, rest '__OTHER__'),
    - splits into train/(val)/test,
    - fits ColumnTransformer on train,
    - optionally applies SMOTE on training features only,
    - returns arrays, transformers, label encoders and optionally balanced_df for inspection.
    """

    # Ensure attack_category exists
    if "attack_category" not in df.columns:
        raise ValueError("run map_attack_category first to create 'attack_category' column.")

    # detect categorical columns if not passed
    if categorical_cols is None:
        categorical_cols = [c for c in ["protocol_type", "service", "flag"] if c in df.columns]
        print("Auto-detected categorical cols:", categorical_cols)

    # convert numeric cols to numeric dtype, fill NaNs temporarily (imputer will handle)
    exclude = set(categorical_cols + ["label", "attack_type", "attack_category", "difficulty_level"])
    numeric_cols = [c for c in df.columns if c not in exclude]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # reduce cardinality on categorical columns (inplace on a df copy)
    df_proc = df.copy()
    for cat in categorical_cols:
        if cat in df_proc.columns:
            df_proc[cat] = reduce_cardinality(df_proc[cat], top_k=rare_topk)

    # train/test split by rows (stratify by attack_category)
    X = df_proc.drop(columns=["attack_type", "attack_category"], errors='ignore')
    y_raw = df_proc["attack_category"].astype(str).values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_raw, test_size=test_size, random_state=random_state, stratify=y_raw
    )
    X_train = X_train_val
    y_train = y_train_val
    X_val = None
    y_val = None

    # optional validation split
    if val_size is not None and val_size > 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )

    # build transformer and fit on training dataframe (important: fit only on train)
    transformer, numeric_cols, categorical_cols = build_preprocessing_pipeline(
        X_train, categorical_cols=categorical_cols, rare_thresh=rare_topk
    )
    transformer.fit(X_train)

    # get transformed feature names (best-effort)
    feature_names = []
    # numeric names
    feature_names.extend(numeric_cols)
    # categorical names (if ohe)
    try:
        ohe = transformer.named_transformers_['cat'].named_steps['ohe']
        cat_feats = list(ohe.get_feature_names_out(categorical_cols))
        feature_names.extend(cat_feats)
    except Exception:
        # fallback: unknown encoder / no categorical cols
        pass

    # transform datasets
    X_train_trans = transformer.transform(X_train)
    X_test_trans = transformer.transform(X_test)
    X_val_trans = transformer.transform(X_val) if X_val is not None else None

    # label encode y (fit on train labels only)
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    y_val_enc = le.transform(y_val) if y_val is not None else None

    # apply SMOTE only on training data if requested
    balanced_info = None
    if apply_smote:
        print("Applying SMOTE on training set only (no leakage).")
        smote = SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)
        X_train_trans, y_train_enc = smote.fit_resample(X_train_trans, y_train_enc)
        balanced_info = Counter(y_train_enc)
        print("New training class distribution:", {le.inverse_transform([k])[0]: v for k, v in balanced_info.items()})

    # compute class weights from original training distribution (before SMOTE typically)
    class_weights = dict()
    try:
        # use y_train (before SMOTE) for class weight calculation, encode to integers using le
        base_y_train_enc = le.transform(y_train)
        classes = np.unique(base_y_train_enc)
        cw = compute_class_weight("balanced", classes=classes, y=base_y_train_enc)
        class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
    except Exception as e:
        print("Could not compute class weights:", e)
        class_weights = {}

    # Optionally save the pipeline
    if save_pipeline_path:
        os.makedirs(os.path.dirname(save_pipeline_path), exist_ok=True)
        joblib.dump({
            "transformer": transformer,
            "label_encoder": le,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_names": feature_names
        }, save_pipeline_path)
        print("Saved preprocessing pipeline to:", save_pipeline_path)

    # wrap outputs in a dictionary for readability
    out = {
        "X_train": X_train_trans, "y_train": y_train_enc,
        "X_val": X_val_trans, "y_val": y_val_enc,
        "X_test": X_test_trans, "y_test": y_test_enc,
        "transformer": transformer, "label_encoder": le,
        "feature_names": feature_names, "class_weights": class_weights,
        "balanced_info": balanced_info,
        # original dataframes for inspection
        "X_train_raw": X_train.reset_index(drop=True),
        "y_train_raw": y_train,
        "X_test_raw": X_test.reset_index(drop=True),
        "y_test_raw": y_test
    }

    print("Preprocessing complete. Shapes:")
    print("  X_train:", out["X_train"].shape)
    if out["X_val"] is not None:
        print("  X_val:  ", out["X_val"].shape)
    print("  X_test: ", out["X_test"].shape)
    return out


# ---------------- quick usage example ----------------
if __name__ == "__main__":
    # Example run (change paths accordingly)
    df = load_nsl_kdd("data/NSL-KDD/KDDTrain+.txt")
    df = assign_column_names_if_possible(df)
    df = map_attack_category(df, label_col="label")

    prepared = prepare_data_pipeline(
        df,
        categorical_cols=None,   # autodetect
        test_size=0.2,
        val_size=0.1,            # optional validation split
        random_state=42,
        apply_smote=True,        # SMOTE only on train
        smote_k_neighbors=5,
        rare_topk=40,            # keep top-40 categories for service, rest -> __OTHER__
        save_pipeline_path="artifacts/preproc_pipeline.joblib"
    )

    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_val, y_val = prepared["X_val"], prepared["y_val"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]

    print("Example feature names (first 20):", prepared["feature_names"][:20])
    print("Class weights (for model.fit):", prepared["class_weights"])
    if prepared["balanced_info"] is not None:
        print("Balanced training distribution (post-SMOTE):", prepared["balanced_info"])
