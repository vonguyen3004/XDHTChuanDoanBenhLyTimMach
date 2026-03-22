import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from sklearn.metrics import accuracy_score, classification_report, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Bidirectional, Concatenate, Conv1D, Dense, Dropout, GlobalAveragePooling1D, Input, LSTM, MaxPooling1D, Multiply, Reshape, SpatialDropout1D
from tensorflow.keras.models import Model, load_model


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "ptbxl"
MODEL_DIR = Path(__file__).resolve().parent
CONFIG_DIR = MODEL_DIR.parent / "config"
MODEL_PATH = MODEL_DIR / "cnn_model.keras"
LABEL_PATH = CONFIG_DIR / "label_names.json" if (CONFIG_DIR / "label_names.json").exists() else MODEL_DIR / "label_names.json"
THRESHOLD_PATH = CONFIG_DIR / "label_thresholds.json" if (CONFIG_DIR / "label_thresholds.json").exists() else MODEL_DIR / "label_thresholds.json"
PTBXL_SIGNAL_COLUMN = "filename_hr"
PTBXL_SAMPLING_RATE = 500
ECG_TARGET_LEN = 5000
HISTORY_PATH = MODEL_DIR / "training_history.json"
REPORT_PATH = MODEL_DIR / "classification_report.json"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.json"
AUG_NOISE_STD = 0.004
AUG_SCALE_MIN = 0.95
AUG_SCALE_MAX = 1.05
FOCAL_GAMMA = 1.5
APPLY_TRAIN_AUGMENT = True

_MODEL_CACHE = None
_LABEL_CACHE = None


TARGET_DISEASES = [
    "NORM",
    "MI",
    "AMI",
    "IMI",
    "ASMI",
    "ALMI",
    "LMI",
    "PMI",
    "STTC",
    "HYP",
    "LVH",
    "RVH",
    "LAH",
    "RAH",
    "CD",
    "RBBB",
    "LBBB",
    "AFIB",
    "PAC",
    "PVC",
]


TARGET_CODE_MAP: Dict[str, List[str]] = {
    "NORM": ["NORM"],
    "MI": ["MI", "AMI", "IMI", "ASMI", "ALMI", "LMI", "PMI", "IPMI", "ILMI", "IPLMI"],
    "AMI": ["AMI", "ASMI", "ALMI"],
    "IMI": ["IMI", "IPMI", "ILMI", "IPLMI"],
    "ASMI": ["ASMI"],
    "ALMI": ["ALMI"],
    "LMI": ["LMI", "ILMI", "IPLMI"],
    "PMI": ["PMI", "IPMI", "IPLMI"],
    "STTC": ["STTC", "NDT", "NST_", "ISC_", "ISCAL", "ISCAS", "ISCIL", "ISCIN", "ISCLA", "ISCAN", "DIG", "LNGQT", "ANEUR", "EL"],
    "HYP": ["HYP", "LVH", "RVH", "SEHYP", "LAO/LAE", "RAO/RAE"],
    "LVH": ["LVH", "VCLVH", "SEHYP"],
    "RVH": ["RVH"],
    "LAH": ["LAO/LAE", "LAH"],
    "RAH": ["RAO/RAE", "RAH"],
    "CD": ["CD", "IRBBB", "CRBBB", "ILBBB", "CLBBB", "LAFB", "LPFB", "IVCD", "1AVB", "2AVB", "3AVB", "WPW"],
    "RBBB": ["RBBB", "IRBBB", "CRBBB"],
    "LBBB": ["LBBB", "ILBBB", "CLBBB"],
    "AFIB": ["AFIB"],
    "PAC": ["PAC", "SVES"],
    "PVC": ["PVC", "VES"],
}


def parse_scp_codes(scp_codes: str) -> Dict[str, float]:
    if pd.isna(scp_codes):
        return {}
    return ast.literal_eval(scp_codes)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std


def to_fixed_shape(signal: np.ndarray, target_len: int = ECG_TARGET_LEN, n_leads: int = 12) -> np.ndarray:
    if signal.shape[1] != n_leads:
        if signal.shape[1] > n_leads:
            signal = signal[:, :n_leads]
        else:
            pad = np.zeros((signal.shape[0], n_leads - signal.shape[1]), dtype=signal.dtype)
            signal = np.concatenate([signal, pad], axis=1)

    if signal.shape[0] == target_len:
        return signal
    if signal.shape[0] > target_len:
        return signal[:target_len]

    pad_len = target_len - signal.shape[0]
    pad_block = np.zeros((pad_len, n_leads), dtype=signal.dtype)
    return np.concatenate([signal, pad_block], axis=0)


def load_ptbxl_metadata(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db = pd.read_csv(data_dir / "ptbxl_database.csv")
    scp_statements = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    db["scp_codes_parsed"] = db["scp_codes"].apply(parse_scp_codes)
    return db, scp_statements


def build_multilabel_vector(scp_code_dict: Dict[str, float], scp_map_df: pd.DataFrame) -> np.ndarray:
    code_set = set(scp_code_dict.keys())

    # Expand with diagnostic classes from scp_statements for broad labels like MI/STTC/HYP/CD.
    for code in list(code_set):
        if code in scp_map_df.index:
            diag_class = scp_map_df.at[code, "diagnostic_class"]
            if isinstance(diag_class, str) and diag_class:
                code_set.add(diag_class)

    label = np.zeros(len(TARGET_DISEASES), dtype=np.float32)
    for i, disease in enumerate(TARGET_DISEASES):
        aliases = TARGET_CODE_MAP[disease]
        if any(alias in code_set for alias in aliases):
            label[i] = 1.0
    return label


def build_record_index(data_dir: Path, max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    db, scp_map_df = load_ptbxl_metadata(data_dir)

    record_list: List[str] = []
    y_list: List[np.ndarray] = []

    for _, row in db.iterrows():
        label_vec = build_multilabel_vector(row["scp_codes_parsed"], scp_map_df)
        if label_vec.sum() == 0:
            continue

        record_rel_path = row[PTBXL_SIGNAL_COLUMN]
        record_abs_path = data_dir / record_rel_path
        record_list.append(str(record_abs_path))
        y_list.append(label_vec)

        if max_samples > 0 and len(record_list) >= max_samples:
            break

    X = np.asarray(record_list, dtype=str)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y


def residual_conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dropout_rate: float,
) -> tf.Tensor:
    shortcut = x

    x = Conv1D(filters, kernel_size=kernel_size, padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters, kernel_size=3, padding="same", activation=None)(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = SpatialDropout1D(dropout_rate)(x)
    return x


def apply_lead_attention(x: tf.Tensor) -> tf.Tensor:
    # Project feature map to 12 pseudo-lead channels then learn lead-wise gates.
    lead_map = Conv1D(12, kernel_size=1, padding="same", activation="linear")(x)
    lead_context = GlobalAveragePooling1D()(lead_map)
    lead_weights = Dense(12, activation="sigmoid", name="lead_attention_weights")(lead_context)
    lead_weights = Reshape((1, 12))(lead_weights)
    return Multiply(name="lead_attention_multiply")([lead_map, lead_weights])


def residual_head_block(x: tf.Tensor, filters: int = 256) -> tf.Tensor:
    shortcut = x

    x = Conv1D(filters, kernel_size=3, padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters, kernel_size=3, padding="same", activation=None)(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same", activation=None)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    return Activation("relu")(x)


def build_model(input_shape: Tuple[int, int], n_classes: int) -> tf.keras.Model:
    inputs = Input(shape=input_shape)

    # Multi-scale temporal front-end for short/mid/long ECG patterns.
    x_k5 = Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x_k11 = Conv1D(64, kernel_size=11, padding="same", activation="relu")(inputs)
    x_k25 = Conv1D(64, kernel_size=25, padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x_k5, x_k11, x_k25])
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = SpatialDropout1D(0.2)(x)

    x = residual_conv_block(x, filters=128, kernel_size=7, dropout_rate=0.2)
    x = residual_conv_block(x, filters=256, kernel_size=5, dropout_rate=0.25)
    x = residual_conv_block(x, filters=256, kernel_size=5, dropout_rate=0.3)
    x = residual_conv_block(x, filters=512, kernel_size=3, dropout_rate=0.35)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = apply_lead_attention(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation="sigmoid", dtype="float32")(x)

    return Model(inputs=inputs, outputs=outputs, name="ptbxl_cnn_lead_attention")


def compute_positive_class_weights(y_train: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    weights = (neg + eps) / (pos + eps)
    return np.clip(weights, 1.0, 30.0).astype(np.float32)


def build_weighted_bce(pos_weights: np.ndarray):
    pw = tf.constant(pos_weights, dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true_f, y_pred_f)
        weight_map = y_true_f * pw + (1.0 - y_true_f)
        return tf.reduce_mean(bce * weight_map)

    return weighted_bce


def build_weighted_focal_bce(pos_weights: np.ndarray, gamma: float = FOCAL_GAMMA):
    pw = tf.constant(pos_weights, dtype=tf.float32)
    gamma_t = tf.constant(gamma, dtype=tf.float32)

    def weighted_focal_bce(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
        bce = tf.keras.backend.binary_crossentropy(y_true_f, y_pred_f)
        weight_map = y_true_f * pw + (1.0 - y_true_f)
        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        focal_factor = tf.pow(1.0 - p_t, gamma_t)
        return tf.reduce_mean(bce * weight_map * focal_factor)

    return weighted_focal_bce


def load_signal_from_file(path: str) -> np.ndarray:
    signal, _ = wfdb.rdsamp(path)
    signal = to_fixed_shape(signal, target_len=ECG_TARGET_LEN, n_leads=12)
    signal = normalize_signal(signal).astype(np.float32)
    return signal


def _tf_load_signal(path_tensor: tf.Tensor) -> tf.Tensor:
    def _py_load(path_bytes):
        path = path_bytes.numpy().decode("utf-8")
        return load_signal_from_file(path)

    signal = tf.py_function(func=_py_load, inp=[path_tensor], Tout=tf.float32)
    signal.set_shape((ECG_TARGET_LEN, 12))
    return signal


def _tf_augment_signal(signal: tf.Tensor) -> tf.Tensor:
    signal = tf.cast(signal, tf.float32)

    # Add small jitter noise.
    noise = tf.random.normal(tf.shape(signal), mean=0.0, stddev=AUG_NOISE_STD, dtype=tf.float32)
    signal = signal + noise

    # Random amplitude scaling.
    scale = tf.random.uniform([], minval=AUG_SCALE_MIN, maxval=AUG_SCALE_MAX, dtype=tf.float32)
    signal = signal * scale

    # Small temporal roll for robustness against beat misalignment.
    shift = tf.random.uniform([], minval=-25, maxval=26, dtype=tf.int32)
    signal = tf.roll(signal, shift=shift, axis=0)

    return signal


def make_tf_dataset(
    paths: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, yy: (_tf_load_signal(p), yy),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if training and APPLY_TRAIN_AUGMENT:
        ds = ds.map(
            lambda x, yy: (_tf_augment_signal(x), yy),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_inference_dataset(paths: np.ndarray, batch_size: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_tf_load_signal, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_training_artifacts(
    history: tf.keras.callbacks.History,
    y_test: np.ndarray,
    y_pred: np.ndarray,
):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=True, indent=2)

    report = classification_report(y_test, y_pred, target_names=TARGET_DISEASES, zero_division=0, output_dict=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    cm = multilabel_confusion_matrix(y_test, y_pred)
    cm_dict = {label: cm[i].tolist() for i, label in enumerate(TARGET_DISEASES)}
    with open(CONFUSION_PATH, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, ensure_ascii=True, indent=2)


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Coordinate search for per-label thresholds maximising validation micro-F1."""
    grid = np.arange(0.10, 0.91, 0.02)
    thresholds = np.full(len(TARGET_DISEASES), 0.5, dtype=np.float32)

    best_micro = f1_score(
        y_true,
        (y_prob >= thresholds.reshape(1, -1)).astype(np.int32),
        average="micro",
        zero_division=0,
    )

    for _ in range(2):
        improved = False
        for i in range(len(TARGET_DISEASES)):
            current_best_th = float(thresholds[i])
            current_best_micro = best_micro
            for th in grid:
                trial = thresholds.copy()
                trial[i] = th
                pred = (y_prob >= trial.reshape(1, -1)).astype(np.int32)
                micro = f1_score(y_true, pred, average="micro", zero_division=0)
                if micro > current_best_micro:
                    current_best_micro = micro
                    current_best_th = float(th)
            if current_best_micro > best_micro:
                thresholds[i] = current_best_th
                best_micro = current_best_micro
                improved = True
        if not improved:
            break

    print(f"Threshold tuning (val micro-F1): {best_micro:.4f}")
    return {label: float(thresholds[i]) for i, label in enumerate(TARGET_DISEASES)}


class WarmupLearningRate(Callback):
    def __init__(self, warmup_epochs: int = 5, start_lr: float = 1e-5, target_lr: float = 1e-3):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.warmup_epochs:
            return
        progress = float(epoch + 1) / float(self.warmup_epochs)
        lr = self.start_lr + progress * (self.target_lr - self.start_lr)
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))


def apply_thresholds(y_prob: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    out = np.zeros_like(y_prob, dtype=np.int32)
    for i, label in enumerate(TARGET_DISEASES):
        th = float(thresholds.get(label, 0.5))
        out[:, i] = (y_prob[:, i] >= th).astype(np.int32)
    return out


def train_and_evaluate(data_dir: Path = DEFAULT_DATA_DIR, epochs: int = 70, batch_size: int = 64, max_samples: int = 0):
    np.random.seed(42)
    tf.random.set_seed(42)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Detected GPUs: {gpus}")
    if gpus:
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    print("Building record index from PTB-XL...")
    X, y = build_record_index(data_dir=data_dir, max_samples=max_samples)

    if len(X) == 0:
        raise RuntimeError("No valid ECG samples found after filtering target diseases.")

    print(f"Dataset size: n_records={len(X)}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )

    model = build_model(input_shape=(ECG_TARGET_LEN, 12), n_classes=len(TARGET_DISEASES))
    pos_weights = compute_positive_class_weights(y_train)
    class_weight_map = {label: float(pos_weights[i]) for i, label in enumerate(TARGET_DISEASES)}
    print("Positive class weights:", class_weight_map)

    train_ds = make_tf_dataset(
        X_train,
        y_train,
        batch_size=batch_size,
        training=True,
    )
    val_ds = make_tf_dataset(
        X_val,
        y_val,
        batch_size=batch_size,
        training=False,
    )
    test_ds = make_inference_dataset(X_test, batch_size=batch_size)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=build_weighted_focal_bce(pos_weights),
        metrics=["accuracy"],
    )

    callbacks = [
        WarmupLearningRate(warmup_epochs=5, start_lr=1e-5, target_lr=1e-3),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        TerminateOnNaN(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    if MODEL_PATH.exists():
        # compile=False avoids deserialization issues with custom training loss.
        model = load_model(MODEL_PATH, compile=False)
    else:
        model.save(MODEL_PATH)

    y_val_input_ds = make_inference_dataset(X_val, batch_size=batch_size)
    y_val_proba = model.predict(y_val_input_ds, verbose=0)
    best_thresholds = optimize_thresholds(y_val, y_val_proba)

    y_proba = model.predict(test_ds, verbose=0)
    y_pred = apply_thresholds(y_proba, best_thresholds)

    acc = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    _PREV_MICRO_F1 = 0.7331
    delta = f1_micro - _PREV_MICRO_F1
    print(f"Test Exact-Match Accuracy: {acc:.4f}")
    print(f"Test F1-score (micro): {f1_micro:.4f}  (prev={_PREV_MICRO_F1:.4f}, delta={delta:+.4f})")
    print(f"Test F1-score (macro): {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, target_names=TARGET_DISEASES, zero_division=0))

    save_training_artifacts(history, y_test, y_pred)

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        json.dump(TARGET_DISEASES, f, ensure_ascii=True, indent=2)

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(best_thresholds, f, ensure_ascii=True, indent=2)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved labels to: {LABEL_PATH}")
    print(f"Saved thresholds to: {THRESHOLD_PATH}")
    print(f"Saved training history to: {HISTORY_PATH}")
    print(f"Saved classification report to: {REPORT_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_PATH}")

    return model


def load_inference_artifacts(model_path: Path = MODEL_PATH, label_path: Path = LABEL_PATH):
    global _MODEL_CACHE, _LABEL_CACHE

    if _MODEL_CACHE is None:
        # compile=False keeps inference independent from custom training loss definition.
        _MODEL_CACHE = load_model(model_path, compile=False)

    if _LABEL_CACHE is None:
        with open(label_path, "r", encoding="utf-8") as f:
            _LABEL_CACHE = json.load(f)

    return _MODEL_CACHE, _LABEL_CACHE


def prepare_signal_array(signal: np.ndarray | List[List[float]] | List[float]) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)

    if arr.ndim == 1:
        if arr.size % 12 == 0:
            arr = arr.reshape((-1, 12))
        else:
            raise ValueError("1D signal length must be divisible by 12 to infer ECG leads.")
    elif arr.ndim != 2:
        raise ValueError("Signal must be 1D or 2D array-like.")

    arr = to_fixed_shape(arr, target_len=ECG_TARGET_LEN, n_leads=12)
    arr = normalize_signal(arr).astype(np.float32)
    return arr


def predict_ecg_array(
    signal: np.ndarray | List[List[float]] | List[float],
    threshold: float | Dict[str, float] = 0.5,
    model_path: Path = MODEL_PATH,
    model: tf.keras.Model | None = None,
    label_names: List[str] | None = None,
) -> List[Dict[str, float]]:
    if model is None or label_names is None:
        model, label_names = load_inference_artifacts(model_path=model_path, label_path=LABEL_PATH)

    arr = prepare_signal_array(signal)
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]

    if threshold == 0.5 and THRESHOLD_PATH.exists():
        try:
            with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
                threshold = json.load(f)
        except Exception:
            threshold = 0.5

    predictions: List[Dict[str, float]] = []
    for label, prob in zip(label_names, probs):
        if isinstance(threshold, dict):
            th = float(threshold.get(label, 0.5))
        else:
            th = float(threshold)

        if prob >= th:
            predictions.append({"disease": label, "confidence": float(prob)})

    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions


def predict_ecg(
    file_path: str,
    threshold: float | Dict[str, float] = 0.5,
    model_path: Path = MODEL_PATH,
    model: tf.keras.Model | None = None,
    label_names: List[str] | None = None,
) -> List[Dict[str, float]]:
    if model is None or label_names is None:
        model, label_names = load_inference_artifacts(model_path=model_path, label_path=LABEL_PATH)

    signal, _ = wfdb.rdsamp(file_path)
    return predict_ecg_array(
        signal=signal,
        threshold=threshold,
        model_path=model_path,
        model=model,
        label_names=label_names,
    )


if __name__ == "__main__":
    # Adjust max_samples>0 for quick debug runs on a smaller subset.
    train_and_evaluate(data_dir=DEFAULT_DATA_DIR, epochs=70, batch_size=64, max_samples=0)
