from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, flash
import os
import io
import datetime
import sys
import tempfile
import json
from collections import Counter
import numpy as np
import joblib
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
from pathlib import Path
import wfdb
# weasyprint removed due to Windows libraries

# configuration
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = 'ecgdb'

# serve static files from the backend/static directory (was previously ../frontend)
app = Flask(__name__, static_folder='static', template_folder='templates')
# required for session/flash
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-for-local')

# Jinja2 filter: format datetime values as "HH:MM:SS DD/MM/YYYY"
def format_datetime(value):
    """Format a datetime-like value as local VN time: HH:MM:SS DD/MM/YYYY.

    - If the stored value is naive, assume it's UTC.
    - If it's a string, try parsing several common formats.
    - Convert to Asia/Ho_Chi_Minh timezone before formatting.
    """
    if value is None:
        return ''
    # resolve to datetime
    val = None
    if isinstance(value, str):
        for fmt in (None, "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                if fmt is None:
                    # fromisoformat handles many ISO forms
                    val = datetime.datetime.fromisoformat(value)
                else:
                    val = datetime.datetime.strptime(value, fmt)
                break
            except Exception:
                val = None
        if val is None:
            return value
    elif isinstance(value, datetime.datetime):
        val = value
    else:
        try:
            val = datetime.datetime.fromtimestamp(float(value))
        except Exception:
            return str(value)

    # timezone handling: treat naive datetimes as UTC, convert to VN
    try:
        from zoneinfo import ZoneInfo
        vn_tz = ZoneInfo('Asia/Ho_Chi_Minh')
    except Exception:
        vn_tz = None

    if val.tzinfo is None:
        # assume stored timestamps are UTC
        try:
            val = val.replace(tzinfo=datetime.timezone.utc)
        except Exception:
            pass

    if vn_tz is not None:
        try:
            val = val.astimezone(vn_tz)
        except Exception:
            # if conversion fails, ignore
            pass
    else:
        # no zoneinfo available, apply fixed offset +7
        try:
            val = val.astimezone(datetime.timezone(datetime.timedelta(hours=7)))
        except Exception:
            pass

    return val.strftime("%H:%M:%S %d/%m/%Y")

# register filter early so templates compiled later can use it
app.jinja_env.filters['datetimeformat'] = format_datetime

# --- database setup ------------------------------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
ecgs_collection = db.ecgs
settings_collection = db.settings
patients_collection = db.patients
ecg_records_collection = db.ecg_records

# load model if available
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml', 'RandomForestClassifier_model', 'RandomForestClassifier_model.joblib'))
FALLBACK_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
model = None

ROOT_DIR = Path(__file__).resolve().parents[1]
CNN_MODEL_PATH = ROOT_DIR / 'ml' / 'cnn_model' / 'cnn_model.keras'
CNN_DIR = ROOT_DIR / 'ml' / 'cnn_model'
CNN_LABEL_PATH = (
    CNN_DIR / 'config' / 'label_names.json'
    if (CNN_DIR / 'config' / 'label_names.json').exists()
    else CNN_DIR / 'label_names.json'
)
cnn_predict_ecg = None
cnn_predict_ecg_array = None
cnn_model = None
cnn_labels = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

if model is None and os.path.exists(FALLBACK_MODEL_PATH):
    try:
        model = joblib.load(FALLBACK_MODEL_PATH)
    except Exception:
        model = None

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if CNN_MODEL_PATH.exists() and CNN_LABEL_PATH.exists():
    try:
        from ml.cnn_model.scripts.train_cnn import load_inference_artifacts, predict_ecg, predict_ecg_array

        cnn_model, cnn_labels = load_inference_artifacts(model_path=CNN_MODEL_PATH, label_path=CNN_LABEL_PATH)
        cnn_predict_ecg = predict_ecg
        cnn_predict_ecg_array = predict_ecg_array
    except Exception as e:
        print('CNN model load failed:', e)
        cnn_model = None
        cnn_labels = None
        cnn_predict_ecg = None
        cnn_predict_ecg_array = None


DISEASE_VN = {
    'NORM': 'Điện tâm đồ bình thường',
    'MI': 'Nhồi máu cơ tim',
    'AMI': 'Nhồi máu cơ tim cấp',
    'IMI': 'Nhồi máu cơ tim thành dưới',
    'ASMI': 'Nhồi máu cơ tim trước vách',
    'ALMI': 'Nhồi máu cơ tim trước bên',
    'LMI': 'Nhồi máu cơ tim thành bên',
    'PMI': 'Nhồi máu cơ tim thành sau',
    'STTC': 'Bất thường đoạn ST/T',
    'HYP': 'Phì đại tim',
    'LVH': 'Phì đại thất trái',
    'RVH': 'Phì đại thất phải',
    'LAH': 'Phì đại nhĩ trái',
    'RAH': 'Phì đại nhĩ phải',
    'CD': 'Rối loạn dẫn truyền',
    'RBBB': 'Block nhánh phải',
    'LBBB': 'Block nhánh trái',
    'AFIB': 'Rung nhĩ',
    'PAC': 'Ngoại tâm thu nhĩ',
    'PVC': 'Ngoại tâm thu thất',
}

DISEASE_EN = {
    'NORM': 'Normal ECG',
    'MI': 'Myocardial Infarction',
    'AMI': 'Acute Myocardial Infarction',
    'IMI': 'Inferior Myocardial Infarction',
    'ASMI': 'Anteroseptal Myocardial Infarction',
    'ALMI': 'Anterolateral Myocardial Infarction',
    'LMI': 'Lateral Myocardial Infarction',
    'PMI': 'Posterior Myocardial Infarction',
    'STTC': 'ST/T Abnormality',
    'HYP': 'Cardiac Hypertrophy',
    'LVH': 'Left Ventricular Hypertrophy',
    'RVH': 'Right Ventricular Hypertrophy',
    'LAH': 'Left Atrial Hypertrophy',
    'RAH': 'Right Atrial Hypertrophy',
    'CD': 'Conduction Disturbance',
    'RBBB': 'Right Bundle Branch Block',
    'LBBB': 'Left Bundle Branch Block',
    'AFIB': 'Atrial Fibrillation',
    'PAC': 'Premature Atrial Contraction',
    'PVC': 'Premature Ventricular Contraction',
}

DISEASE_DETAILS = {
    'NORM': {
        'description': 'Điện tâm đồ bình thường cho thấy hoạt động điện học của tim diễn ra ổn định và đồng bộ. Các sóng P, phức bộ QRS và sóng T có hình dạng, biên độ và khoảng thời gian trong giới hạn sinh lý. Không ghi nhận dấu hiệu thiếu máu cơ tim, rối loạn dẫn truyền hay loạn nhịp.',
        'ecgSigns': 'Nhịp xoang đều, sóng P trước mỗi QRS, khoảng PR và QT bình thường, không có ST chênh hoặc sóng Q bệnh lý.',
        'severity': 'Không có',
        'recommendation': 'Không cần can thiệp. Duy trì lối sống lành mạnh, kiểm tra định kỳ để theo dõi sức khỏe tim mạch.'
    },
    'MI': {
        'description': 'Nhồi máu cơ tim xảy ra khi dòng máu nuôi cơ tim bị tắc nghẽn kéo dài, gây hoại tử mô tim. Đây là một trong những bệnh lý tim mạch nguy hiểm, có thể dẫn đến suy tim hoặc tử vong nếu không điều trị kịp thời.',
        'ecgSigns': 'ST chênh lên hoặc xuống, xuất hiện sóng Q bệnh lý, đảo ngược sóng T, thay đổi theo vùng tổn thương.',
        'severity': 'Cao',
        'recommendation': 'Cần nhập viện để đánh giá và điều trị sớm, tránh biến chứng nguy hiểm.'
    },
    'AMI': {
        'description': 'Nhồi máu cơ tim cấp là tình trạng cấp cứu do tắc hoàn toàn động mạch vành, gây tổn thương nghiêm trọng cơ tim trong thời gian ngắn.',
        'ecgSigns': 'ST chênh lên rõ rệt ở các chuyển đạo liên quan, có thể kèm rối loạn nhịp.',
        'severity': 'Rất cao',
        'recommendation': 'Cần cấp cứu ngay lập tức và can thiệp mạch vành khẩn cấp.'
    },
    'IMI': {
        'description': 'Nhồi máu cơ tim thành dưới ảnh hưởng vùng dưới của tim, thường liên quan động mạch vành phải.',
        'ecgSigns': 'ST chênh lên ở II, III, aVF, có thể kèm nhịp chậm.',
        'severity': 'Cao',
        'recommendation': 'Cần theo dõi sát và điều trị chuyên khoa tim mạch.'
    },
    'ASMI': {
        'description': 'Nhồi máu cơ tim trước vách ảnh hưởng vùng trước và vách liên thất, có thể làm suy giảm chức năng co bóp của tim.',
        'ecgSigns': 'ST chênh lên ở V1–V4, sóng Q có thể xuất hiện.',
        'severity': 'Cao',
        'recommendation': 'Nguy cơ suy tim cao, cần can thiệp sớm.'
    },
    'ALMI': {
        'description': 'Nhồi máu cơ tim trước bên ảnh hưởng vùng trước và bên của thất trái.',
        'ecgSigns': 'Bất thường ở I, aVL, V4–V6.',
        'severity': 'Cao',
        'recommendation': 'Cần theo dõi và điều trị tích cực.'
    },
    'LMI': {
        'description': 'Nhồi máu cơ tim thành bên gây tổn thương vùng bên của tim.',
        'ecgSigns': 'Thay đổi ST-T ở I, aVL, V5–V6.',
        'severity': 'Cao',
        'recommendation': 'Có thể ảnh hưởng chức năng thất trái, cần điều trị.'
    },
    'PMI': {
        'description': 'Nhồi máu cơ tim thành sau thường khó phát hiện do không có chuyển đạo trực tiếp trên ECG tiêu chuẩn.',
        'ecgSigns': 'ST chênh xuống ở V1–V3, sóng R cao bất thường.',
        'severity': 'Cao',
        'recommendation': 'Cần kết hợp lâm sàng và xét nghiệm bổ sung.'
    },
    'STTC': {
        'description': 'Bất thường đoạn ST/T phản ánh sự thay đổi tái cực cơ tim, có thể do thiếu máu cơ tim, rối loạn điện giải hoặc tác dụng thuốc.',
        'ecgSigns': 'ST chênh nhẹ, sóng T đảo hoặc dẹt.',
        'severity': 'Trung bình',
        'recommendation': 'Cần theo dõi thêm và đánh giá nguyên nhân.'
    },
    'HYP': {
        'description': 'Phì đại tim là tình trạng tim tăng kích thước do phải làm việc quá mức trong thời gian dài.',
        'ecgSigns': 'Điện thế QRS cao, thay đổi trục tim.',
        'severity': 'Trung bình',
        'recommendation': 'Cần điều trị nguyên nhân như tăng huyết áp.'
    },
    'LVH': {
        'description': 'Phì đại thất trái xảy ra khi thất trái phải làm việc nhiều hơn bình thường, thường do tăng huyết áp kéo dài.',
        'ecgSigns': 'QRS cao ở V5–V6, tiêu chuẩn điện thế tăng.',
        'severity': 'Trung bình → Cao',
        'recommendation': 'Cần kiểm soát huyết áp và theo dõi tim mạch.'
    },
    'RVH': {
        'description': 'Phì đại thất phải thường liên quan đến bệnh phổi hoặc tăng áp động mạch phổi.',
        'ecgSigns': 'Trục tim lệch phải, sóng R cao ở V1.',
        'severity': 'Trung bình',
        'recommendation': 'Đánh giá chức năng tim và phổi.'
    },
    'LAH': {
        'description': 'Phì đại nhĩ trái xảy ra khi nhĩ trái giãn hoặc dày lên do tăng áp lực kéo dài.',
        'ecgSigns': 'Sóng P rộng, hai đỉnh (P mitrale).',
        'severity': 'Trung bình',
        'recommendation': 'Theo dõi bệnh nền như tăng huyết áp.'
    },
    'RAH': {
        'description': 'Phì đại nhĩ phải thường liên quan đến bệnh phổi mạn tính.',
        'ecgSigns': 'Sóng P cao, nhọn (P pulmonale).',
        'severity': 'Trung bình',
        'recommendation': 'Cần kiểm tra thêm hệ hô hấp.'
    },
    'CD': {
        'description': 'Rối loạn dẫn truyền xảy ra khi tín hiệu điện tim truyền qua hệ dẫn truyền bị chậm hoặc gián đoạn.',
        'ecgSigns': 'Khoảng PR hoặc QRS kéo dài.',
        'severity': 'Trung bình',
        'recommendation': 'Theo dõi nguy cơ loạn nhịp.'
    },
    'RBBB': {
        'description': 'Block nhánh phải là tình trạng chậm dẫn truyền ở nhánh phải của bó His.',
        'ecgSigns': 'QRS rộng, dạng chữ M ở V1.',
        'severity': 'Thấp → Trung bình',
        'recommendation': 'Thường lành tính nếu không có bệnh nền.'
    },
    'LBBB': {
        'description': 'Block nhánh trái làm mất đồng bộ co bóp thất trái, có thể che lấp dấu hiệu nhồi máu.',
        'ecgSigns': 'QRS rộng, biến dạng phức bộ QRS.',
        'severity': 'Trung bình → Cao',
        'recommendation': 'Cần đánh giá tim mạch kỹ hơn.'
    },
    'AFIB': {
        'description': 'Rung nhĩ là tình trạng nhịp tim không đều do hoạt động điện hỗn loạn ở tâm nhĩ, làm tăng nguy cơ hình thành huyết khối.',
        'ecgSigns': 'Không có sóng P, RR không đều.',
        'severity': 'Cao',
        'recommendation': 'Cần điều trị chống đông để phòng ngừa đột quỵ.'
    },
    'PAC': {
        'description': 'Ngoại tâm thu nhĩ là nhịp tim sớm xuất phát từ nhĩ, thường lành tính.',
        'ecgSigns': 'Sóng P bất thường xuất hiện sớm.',
        'severity': 'Thấp',
        'recommendation': 'Theo dõi, không cần điều trị nếu không triệu chứng.'
    },
    'PVC': {
        'description': 'Ngoại tâm thu thất là nhịp sớm xuất phát từ thất, có thể gây cảm giác hồi hộp hoặc loạn nhịp.',
        'ecgSigns': 'QRS rộng, không có sóng P trước.',
        'severity': 'Trung bình → Cao',
        'recommendation': 'Nếu xuất hiện nhiều cần khám chuyên khoa.'
    }
}

DEFAULT_LABEL_THRESHOLDS = {
    'NORM': 0.5,
    'MI': 0.5,
    'AMI': 0.5,
    'IMI': 0.5,
    'ASMI': 0.5,
    'ALMI': 0.45,
    'LMI': 0.45,
    'PMI': 0.35,
    'STTC': 0.5,
    'HYP': 0.5,
    'LVH': 0.5,
    'RVH': 0.35,
    'LAH': 0.35,
    'RAH': 0.35,
    'CD': 0.5,
    'RBBB': 0.5,
    'LBBB': 0.5,
    'AFIB': 0.5,
    'PAC': 0.35,
    'PVC': 0.45,
}

THRESHOLD_SETTING_KEY = 'cnn_label_thresholds'


# ============ Helper Functions for Disease Details ============

def normalize_label(label):
    """
    Normalize disease label: trim whitespace and convert to uppercase.
    Args:
        label: Disease label string (e.g., ' IMI ', 'imi')
    Returns:
        Normalized label (e.g., 'IMI')
    """
    if not label:
        return ''
    return str(label).strip().upper()


def getDiseaseDetail(label):
    """Return disease detail by normalized label."""
    normalized = normalize_label(label)
    return DISEASE_DETAILS.get(normalized)


def getSeverityColor(severity):
    """Map severity text to CSS badge class."""
    if not severity:
        return 'severity-medium'

    severity_lower = str(severity).lower()
    if 'rất cao' in severity_lower:
        return 'severity-critical'
    if 'cao' in severity_lower:
        return 'severity-high'
    if 'trung bình' in severity_lower:
        return 'severity-medium'
    if 'thấp' in severity_lower:
        return 'severity-low'
    return 'severity-soft'


def getVietnameseName(label):
    """Return Vietnamese disease name from label."""
    return DISEASE_VN.get(normalize_label(label), '')


def getEnglishName(label):
    """Return English disease name from label."""
    return DISEASE_EN.get(normalize_label(label), '')


# Backward-compatible wrappers
def get_disease_detail(label):
    return getDiseaseDetail(label)


def get_severity_class(severity):
    return getSeverityColor(severity)


@app.context_processor
def inject_diagnosis_helpers():
    return {
        'getDiseaseDetail': getDiseaseDetail,
        'getSeverityColor': getSeverityColor,
        'getVietnameseName': getVietnameseName,
        'getEnglishName': getEnglishName,
        'normalizeLabel': normalize_label,
    }


# ============ End Helper Functions ============


def get_active_thresholds():
    doc = settings_collection.find_one({'key': THRESHOLD_SETTING_KEY})
    if not doc or not isinstance(doc.get('thresholds'), dict):
        return dict(DEFAULT_LABEL_THRESHOLDS)

    out = dict(DEFAULT_LABEL_THRESHOLDS)
    for k, v in doc['thresholds'].items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def save_active_thresholds(thresholds):
    normalized = {}
    for k, v in thresholds.items():
        try:
            normalized[str(k)] = float(v)
        except Exception:
            continue
    settings_collection.update_one(
        {'key': THRESHOLD_SETTING_KEY},
        {
            '$set': {
                'thresholds': normalized,
                'updated_at': datetime.datetime.utcnow(),
            }
        },
        upsert=True,
    )


def get_cnn_model_version():
    try:
        ts = datetime.datetime.utcfromtimestamp(CNN_MODEL_PATH.stat().st_mtime)
        return f"cnn_model.keras@{ts.strftime('%Y%m%d%H%M%S')}"
    except Exception:
        return 'cnn_model.keras@unknown'

# --- utility functions ---------------------------------------------
# detailed descriptions for MIT‑BIH class codes
CLASS_LABELS = {
    0: {'short': 'Bình thường',
        'desc': 'Nhịp tim đều, không có bất thường. Đây là tim khỏe.'},
    1: {'short': 'LBBB',
        'desc': 'Left bundle branch block – tắc nghẽn nhánh trái bó His. Dẫn truyền điện tim bên trái bị chậm, QRS rộng. Có thể liên quan bệnh tim cấu trúc. Rối loạn dẫn truyền điện tim.'},
    2: {'short': 'RBBB',
        'desc': 'Right bundle branch block – tắc nghẽn nhánh phải bó His. Giống LBBB nhưng ở bên phải; QRS kéo dài. Nhiều người không triệu chứng. Cũng là rối loạn dẫn truyền.'},
    3: {'short': 'PVC',
        'desc': 'Premature Ventricular Contraction – ngoại tâm thu thất. Tim co bóp sớm bất thường từ thất; sóng QRS to và rộng. Người bệnh có thể cảm thấy “tim hẫng 1 nhịp”. Rất phổ biến.'},
    4: {'short': 'PAC',
        'desc': 'Premature Atrial Contraction – ngoại tâm thu nhĩ. Tim co sớm từ nhĩ; nhẹ hơn PVC. Có thể gặp ở người stress, thiếu ngủ. Thường không nguy hiểm.'},
}

LEGACY_CLASS_CODES = {
    0: 'NORM',
    1: 'LBBB',
    2: 'RBBB',
    3: 'PVC',
    4: 'PAC',
}


def _legacy_prediction_item(class_id, confidence):
    class_int = int(class_id)
    disease_code = LEGACY_CLASS_CODES.get(class_int, str(class_int))
    info = CLASS_LABELS.get(class_int, {})
    disease_vn = DISEASE_VN.get(disease_code)
    if not disease_vn:
        disease_vn = 'Điện tâm đồ bình thường' if disease_code == 'NORM' else info.get('short', disease_code)
    return {
        'disease': disease_code,
        'disease_vn': disease_vn,
        'confidence': float(confidence),
    }

def preprocess(signal):
    # simple normalization example
    arr = np.array(signal)
    return (arr - arr.mean()) / (arr.std() + 1e-6)


def compute_heart_rate(signal, fs=250):
    # improved peak-based HR with simple refractory filter and capping
    arr = np.array(signal)
    if arr.size < 2:
        # too few samples to have two peaks
        return None
    threshold = arr.mean() + arr.std()
    raw_peaks = np.where(arr > threshold)[0]
    if len(raw_peaks) < 2:
        return None
    # enforce minimum distance between peaks (~0.25s) to avoid noise
    min_samples = int(0.25 * fs)
    peaks = []
    last = -min_samples
    for p in raw_peaks:
        if p - last >= min_samples:
            peaks.append(p)
            last = p
    if len(peaks) < 2:
        return None
    durations = np.diff(peaks) / fs
    avg = np.mean(durations)
    if avg <= 0:
        return None
    hr = 60.0 / avg
    # cap at sensible physiological maximum (e.g. 250 bpm)
    if hr > 250:
        return None
    return float(hr)


def analyze_signal(signal):
    proc = preprocess(signal)
    hr = compute_heart_rate(proc)
    pred = None
    arr = np.array(proc)
    pred_confidence = None
    multi_label_predictions = []
    if model is not None:
        # align to expected feature count if possible
        if hasattr(model, 'n_features_in_'):
            exp = model.n_features_in_
            if arr.size != exp:
                if arr.size == exp + 1:
                    if float(arr[0]).is_integer() and arr[0] in (0, 1, 2, 3, 4):
                        arr = arr[1:]
                    elif float(arr[-1]).is_integer() and arr[-1] in (0, 1, 2, 3, 4):
                        arr = arr[:-1]
                    else:
                        arr = arr[:exp]
                elif arr.size == exp - 1:
                    arr = np.pad(arr, (0, exp - arr.size))
                else:
                    if arr.size <= 1:
                        fill_value = float(arr[0]) if arr.size == 1 else 0.0
                        arr = np.full(exp, fill_value, dtype=np.float32)
                    else:
                        arr = np.interp(
                            np.linspace(0.0, 1.0, exp),
                            np.linspace(0.0, 1.0, arr.size),
                            arr,
                        )
        try:
            pred = model.predict([arr.tolist()])[0]
        except Exception as e:
            print("prediction error", e)
            pred = None
        if pred is not None and hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba([arr.tolist()])[0]
                classes = getattr(model, 'classes_', range(len(probs)))
                multi_label_predictions = sorted(
                    [_legacy_prediction_item(cls, prob) for cls, prob in zip(classes, probs)],
                    key=lambda item: item['confidence'],
                    reverse=True,
                )
                if multi_label_predictions:
                    pred_confidence = float(multi_label_predictions[0]['confidence'])
            except Exception as e:
                print('predict_proba error', e)
    # convert numpy scalars to native Python types for BSON/JSON
    def to_py(val):
        if isinstance(val, np.generic):
            return val.item()
        return val
    result = {'heart_rate': to_py(hr) if hr is not None else None,
              'prediction': to_py(pred)}
    # add label texts if prediction available
    if result['prediction'] is not None:
        info = CLASS_LABELS.get(result['prediction'])
        if info:
            result['prediction_label'] = info['short']
            result['prediction_desc'] = info['desc']
        else:
            result['prediction_label'] = str(result['prediction'])
    if pred_confidence is not None:
        result['prediction_confidence'] = pred_confidence
    if multi_label_predictions:
        result['multi_label_predictions'] = multi_label_predictions
    result['model_type'] = 'random_forest_legacy'
    result['inference_meta'] = {
        'model_type': 'random_forest_legacy',
        'model_version': 'random_forest_legacy',
        'inference_at': datetime.datetime.utcnow().isoformat() + 'Z',
    }
    return result


def _to_2d_signal(signal):
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1 and arr.size % 12 == 0:
        return arr.reshape((-1, 12))
    raise ValueError('signal must be 2D [timesteps, leads] or flattened length divisible by 12')


def _resolve_thresholds(threshold):
    active_defaults = get_active_thresholds()
    if threshold is None:
        return active_defaults
    if isinstance(threshold, dict):
        out = dict(active_defaults)
        for k, v in threshold.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    if isinstance(threshold, str):
        try:
            return float(threshold)
        except Exception:
            return 0.5
    return float(threshold)


def analyze_signal_cnn(signal, threshold=0.5):
    if cnn_predict_ecg_array is None or cnn_model is None or cnn_labels is None:
        raise RuntimeError('CNN model is not available')

    arr2d = _to_2d_signal(signal)
    resolved_threshold = _resolve_thresholds(threshold)
    preds = cnn_predict_ecg_array(
        signal=arr2d,
        threshold=resolved_threshold,
        model=cnn_model,
        label_names=cnn_labels,
    )

    lead_for_hr = arr2d[:, 1] if arr2d.shape[1] > 1 else arr2d[:, 0]
    hr = compute_heart_rate(lead_for_hr, fs=100)

    out = {
        'heart_rate': float(hr) if hr is not None else None,
        'prediction': None,
        'prediction_label': None,
        'prediction_desc': None,
        'prediction_confidence': None,
        'multi_label_predictions': preds,
        'model_type': 'cnn_ptbxl_20',
        'inference_meta': {
            'model_type': 'cnn_ptbxl_20',
            'model_version': get_cnn_model_version(),
            'inference_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'threshold_applied': resolved_threshold,
        },
    }

    if preds:
        top = preds[0]
        out['prediction'] = top['disease']
        out['prediction_label'] = top['disease']
        out['prediction_desc'] = DISEASE_VN.get(top['disease'], top['disease'])
        out['prediction_confidence'] = float(top['confidence'])

    return out


def _load_signal_matrix_from_record_path(record_path_value):
    record_path_str = str(record_path_value or '').strip().strip('"').strip("'")
    if not record_path_str:
        raise ValueError('record_path is empty')

    record_path = Path(record_path_str)
    if record_path.suffix.lower() == '.hea':
        record_path = record_path.with_suffix('')

    if not record_path.is_absolute():
        record_path = ROOT_DIR / record_path

    hea_path = record_path.with_suffix('.hea')
    if not hea_path.exists():
        raise ValueError(f'Không tìm thấy record_path: {record_path_value}')

    signal_matrix, _ = wfdb.rdsamp(str(record_path))
    signal_matrix = np.asarray(signal_matrix, dtype=np.float32)
    if signal_matrix.ndim != 2 or signal_matrix.shape[0] == 0:
        raise ValueError('Record WFDB không chứa tín hiệu ECG hợp lệ')

    return signal_matrix[:, :12]


def _extract_signal_from_uploaded_csv(file_storage):
    raw_bytes = file_storage.read()
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    if not raw_bytes:
        raise ValueError('File tải lên trống')

    try:
        df_named = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        df_named = None

    if df_named is not None and len(df_named.columns) > 0:
        normalized_cols = {str(col).strip().lower(): col for col in df_named.columns}
        if 'record_path' in normalized_cols:
            record_series = df_named[normalized_cols['record_path']].dropna().astype(str).str.strip()
            record_series = record_series[record_series != '']
            if record_series.empty:
                raise ValueError('CSV có cột record_path nhưng không có giá trị hợp lệ')

            record_path_value = record_series.iloc[0]
            signal_matrix = _load_signal_matrix_from_record_path(record_path_value)
            chart_signal = signal_matrix[:, 1] if signal_matrix.shape[1] > 1 else signal_matrix[:, 0]
            return {
                'signal': chart_signal.astype(float).tolist(),
                'signal_matrix': signal_matrix.astype(float).tolist(),
                'source_type': 'csv_record_path',
                'source_ref': record_path_value,
            }

    try:
        df_raw = pd.read_csv(io.BytesIO(raw_bytes), header=None)
    except Exception as e:
        raise ValueError(f'Không đọc được CSV: {e}')

    numeric_df = df_raw.apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    if numeric_df.empty:
        raise ValueError('CSV không chứa dữ liệu ECG dạng số hợp lệ')

    arr = numeric_df.to_numpy(dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    signal_matrix = arr[:, :12] if arr.ndim == 2 and arr.shape[1] >= 12 else None
    if signal_matrix is not None:
        chart_signal = signal_matrix[:, 1] if signal_matrix.shape[1] > 1 else signal_matrix[:, 0]
    else:
        chart_signal = arr.flatten()

    return {
        'signal': np.asarray(chart_signal, dtype=float).tolist(),
        'signal_matrix': signal_matrix.astype(float).tolist() if signal_matrix is not None else None,
        'source_type': 'csv_upload',
        'source_ref': None,
    }


def _analyze_uploaded_payload(signal, signal_matrix=None, threshold=None):
    if signal_matrix is not None and cnn_predict_ecg_array is not None and cnn_model is not None and cnn_labels is not None:
        return analyze_signal_cnn(signal_matrix, threshold=threshold)

    out = analyze_signal(signal)
    if signal_matrix is not None:
        meta = dict(out.get('inference_meta') or {})
        if cnn_predict_ecg_array is None or cnn_model is None or cnn_labels is None:
            meta['fallback_reason'] = 'cnn_model_unavailable'
            meta['fallback_detail'] = 'Sử dụng mô hình dự phòng do CNN chưa sẵn sàng.'
        out['inference_meta'] = meta
    return out

# --- web routes -----------------------------------------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    drift_days_param = request.args.get('drift_days', '').strip()
    if not drift_days_param:
        period_param = request.args.get('period', '').strip()
        if period_param in ('7', '30'):
            drift_days_param = period_param
        else:
            drift_days_param = '14'
    if drift_days_param not in ('7', '14', '30'):
        drift_days_param = '14'
    drift_days = int(drift_days_param)

    drift_top_n_param = request.args.get('drift_top_n', '4').strip()
    if drift_top_n_param not in ('4', '6', '8'):
        drift_top_n_param = '4'
    drift_top_n = int(drift_top_n_param)

    drift_smooth_param = request.args.get('drift_smooth', '0').strip()
    drift_smooth = drift_smooth_param == '1'

    ema_alpha_param = request.args.get('ema_alpha', '0.30').strip()
    try:
        ema_alpha = float(ema_alpha_param)
    except Exception:
        ema_alpha = 0.30
    ema_alpha = min(max(ema_alpha, 0.05), 0.95)

    count = ecgs_collection.count_documents({})
    # compute average heart rate if available
    hr_docs = ecgs_collection.find({'heart_rate': {'$exists': True}}, {'heart_rate': 1})
    hr_list = [d['heart_rate'] for d in hr_docs if d.get('heart_rate') is not None]
    avg_hr = sum(hr_list)/len(hr_list) if hr_list else 0
    # compute uploads per day for last 7 days
    today = datetime.datetime.utcnow().date()
    labels = []
    values = []
    for i in range(6, -1, -1):
        day = today - datetime.timedelta(days=i)
        start = datetime.datetime(day.year, day.month, day.day)
        end = start + datetime.timedelta(days=1)
        num = ecgs_collection.count_documents({'timestamp': {'$gte': start, '$lt': end}})
        labels.append(day.strftime('%d/%m'))
        values.append(num)
    ai_status = 'Online' if (cnn_model is not None or model is not None) else 'Offline'

    today_start = datetime.datetime(today.year, today.month, today.day)
    today_end = today_start + datetime.timedelta(days=1)
    today_uploads = ecgs_collection.count_documents({'timestamp': {'$gte': today_start, '$lt': today_end}})

    recent_docs = ecgs_collection.find(
        {'multi_label_predictions': {'$exists': True}},
        {'multi_label_predictions': 1}
    ).sort('timestamp', -1).limit(300)
    disease_counter = Counter()
    for doc in recent_docs:
        for item in doc.get('multi_label_predictions', []):
            disease = item.get('disease')
            if disease:
                disease_counter[disease] += 1
    top_diseases = [
        {'disease': d, 'count': c, 'vn': DISEASE_VN.get(d, d)}
        for d, c in disease_counter.most_common(5)
    ]

    # Confidence drift by day for top diseases.
    drift_start_day = today - datetime.timedelta(days=drift_days - 1)
    drift_labels = [
        (drift_start_day + datetime.timedelta(days=i)).strftime('%d/%m')
        for i in range(drift_days)
    ]

    drift_candidates = ecgs_collection.find(
        {
            'timestamp': {
                '$gte': datetime.datetime(drift_start_day.year, drift_start_day.month, drift_start_day.day),
                '$lt': datetime.datetime(today.year, today.month, today.day) + datetime.timedelta(days=1),
            },
            'multi_label_predictions': {'$exists': True},
        },
        {'timestamp': 1, 'multi_label_predictions': 1}
    )

    drift_counter = Counter()
    daily_conf = {}
    for i in range(drift_days):
        day = drift_start_day + datetime.timedelta(days=i)
        daily_conf[day.strftime('%Y-%m-%d')] = {}

    for doc in drift_candidates:
        ts = doc.get('timestamp')
        if not isinstance(ts, datetime.datetime):
            continue
        day_key = ts.date().strftime('%Y-%m-%d')
        if day_key not in daily_conf:
            continue

        for item in doc.get('multi_label_predictions', []):
            disease = item.get('disease')
            conf = item.get('confidence')
            if not disease or conf is None:
                continue
            try:
                conf_val = float(conf)
            except Exception:
                continue

            drift_counter[disease] += 1
            bucket = daily_conf[day_key].setdefault(disease, [])
            bucket.append(conf_val * 100.0)

    drift_top_diseases = [d for d, _ in drift_counter.most_common(drift_top_n)]
    drift_series = []

    def compute_ema(values, alpha):
        out = []
        prev = None
        for v in values:
            if v is None:
                out.append(prev)
                continue
            if prev is None:
                prev = v
            else:
                prev = (alpha * v) + ((1.0 - alpha) * prev)
            out.append(round(prev, 2))
        return out

    for disease in drift_top_diseases:
        raw_values = []
        for i in range(drift_days):
            day = drift_start_day + datetime.timedelta(days=i)
            day_key = day.strftime('%Y-%m-%d')
            vals = daily_conf[day_key].get(disease, [])
            if vals:
                raw_values.append(round(sum(vals) / len(vals), 2))
            else:
                raw_values.append(None)

        ema_values = compute_ema(raw_values, ema_alpha)
        selected_values = ema_values if drift_smooth else raw_values

        drift_series.append({
            'disease': disease,
            'vn': DISEASE_VN.get(disease, disease),
            'values': selected_values,
            'raw_values': raw_values,
            'ema_values': ema_values,
        })

    return render_template(
        'dashboard.html',
        count=count,
        avg_hr=avg_hr,
        chart_labels=labels,
        chart_values=values,
        ai_status=ai_status,
        today_uploads=today_uploads,
        top_diseases=top_diseases,
        drift_labels=drift_labels,
        drift_series=drift_series,
        drift_days=drift_days,
        drift_days_param=drift_days_param,
        drift_top_n=drift_top_n,
        drift_top_n_param=drift_top_n_param,
        drift_smooth=drift_smooth,
        drift_smooth_param='1' if drift_smooth else '0',
        ema_alpha=ema_alpha,
        ema_alpha_param=f"{ema_alpha:.2f}",
    )


@app.route('/dashboard/drift-export')
def dashboard_drift_export():
    drift_days_param = request.args.get('drift_days', '').strip()
    if not drift_days_param:
        period_param = request.args.get('period', '').strip()
        if period_param in ('7', '30'):
            drift_days_param = period_param
        else:
            drift_days_param = '14'
    if drift_days_param not in ('7', '14', '30'):
        drift_days_param = '14'
    drift_days = int(drift_days_param)

    drift_top_n_param = request.args.get('drift_top_n', '4').strip()
    if drift_top_n_param not in ('4', '6', '8'):
        drift_top_n_param = '4'
    drift_top_n = int(drift_top_n_param)

    ema_alpha_param = request.args.get('ema_alpha', '0.30').strip()
    try:
        ema_alpha = float(ema_alpha_param)
    except Exception:
        ema_alpha = 0.30
    ema_alpha = min(max(ema_alpha, 0.05), 0.95)

    today = datetime.datetime.utcnow().date()
    drift_start_day = today - datetime.timedelta(days=drift_days - 1)
    start_dt = datetime.datetime(drift_start_day.year, drift_start_day.month, drift_start_day.day)
    end_dt = datetime.datetime(today.year, today.month, today.day) + datetime.timedelta(days=1)

    docs = ecgs_collection.find(
        {
            'timestamp': {'$gte': start_dt, '$lt': end_dt},
            'multi_label_predictions': {'$exists': True},
        },
        {'timestamp': 1, 'multi_label_predictions': 1}
    )

    disease_counter = Counter()
    daily_conf = {}
    for i in range(drift_days):
        day = drift_start_day + datetime.timedelta(days=i)
        daily_conf[day.strftime('%Y-%m-%d')] = {}

    for doc in docs:
        ts = doc.get('timestamp')
        if not isinstance(ts, datetime.datetime):
            continue
        day_key = ts.date().strftime('%Y-%m-%d')
        if day_key not in daily_conf:
            continue
        for item in doc.get('multi_label_predictions', []):
            disease = item.get('disease')
            conf = item.get('confidence')
            if not disease or conf is None:
                continue
            try:
                conf_val = float(conf)
            except Exception:
                continue
            disease_counter[disease] += 1
            bucket = daily_conf[day_key].setdefault(disease, [])
            bucket.append(conf_val * 100.0)

    selected = [d for d, _ in disease_counter.most_common(drift_top_n)]

    def compute_ema(values, alpha):
        out = []
        prev = None
        for v in values:
            if v is None:
                out.append(prev)
                continue
            if prev is None:
                prev = v
            else:
                prev = (alpha * v) + ((1.0 - alpha) * prev)
            out.append(round(prev, 4))
        return out

    import csv
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(['date', 'disease', 'disease_vn', 'avg_confidence_percent', 'ema_confidence_percent', 'n_samples'])

    for i in range(drift_days):
        day = drift_start_day + datetime.timedelta(days=i)
        day_key = day.strftime('%Y-%m-%d')
        for disease in selected:
            vals = daily_conf[day_key].get(disease, [])
            avg_conf = round(sum(vals) / len(vals), 4) if vals else None
            # Recompute disease series EMA in export-friendly way.
            disease_series = []
            for j in range(drift_days):
                d = drift_start_day + datetime.timedelta(days=j)
                dk = d.strftime('%Y-%m-%d')
                v = daily_conf[dk].get(disease, [])
                disease_series.append(round(sum(v) / len(v), 4) if v else None)
            ema_series = compute_ema(disease_series, ema_alpha)
            ema_conf = ema_series[i]
            writer.writerow([
                day_key,
                disease,
                DISEASE_VN.get(disease, disease),
                '' if avg_conf is None else avg_conf,
                '' if ema_conf is None else ema_conf,
                len(vals),
            ])

    output = si.getvalue()
    filename = f'drift_confidence_{drift_days}d.csv'
    return output, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename="{filename}"'
    }

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    import re
    import time
    next_patient_id = request.args.get('next_patient_id')
    # --- Sinh mã bệnh nhân tự động tăng ---
    last_patient = patients_collection.find_one(
        {'patient_id': {'$regex': r'^BN\\d+$'}},
        sort=[('patient_id', -1)]
    )
    last_patient_id = last_patient['patient_id'] if last_patient and 'patient_id' in last_patient else None
    if not next_patient_id:
        next_patient_id = "BN001"
        if last_patient and 'patient_id' in last_patient:
            m = re.match(r'^BN(\d+)$', last_patient['patient_id'])
            if m:
                next_num = int(m.group(1)) + 1
                next_patient_id = f"BN{next_num:03d}"
        t0 = time.time()
        mode = request.form.get('upload_mode', 'csv').strip().lower()
        threshold_form = request.form.get('threshold', '0.5')
        threshold_map_form = request.form.get('threshold_map', '').strip()

        patient_data, patient_error = _parse_patient_form(request.form)
        if patient_error:
            flash(patient_error, 'danger')
            # Truyền lại next_patient_id và last_patient_id khi redirect
            last_patient = patients_collection.find_one(
                {'patient_id': {'$regex': r'^BN\\d+$'}},
                sort=[('patient_id', -1)]
            )
            last_patient_id = last_patient['patient_id'] if last_patient and 'patient_id' in last_patient else None
            return redirect(url_for('upload', next_patient_id=next_patient_id, last_patient_id=last_patient_id))
        _ensure_patient_exists(patient_data)

        threshold = threshold_form
        if threshold_map_form:
            try:
                threshold = json.loads(threshold_map_form)
            except Exception:
                threshold = threshold_form
        elif threshold_form == '' or threshold_form is None:
            threshold = None

        # WFDB upload mode: expect pair .hea + .dat and run CNN inference.
        if mode == 'wfdb':
            hea_file = request.files.get('hea_file')
            dat_file = request.files.get('dat_file')
            if hea_file and dat_file and cnn_predict_ecg is not None:
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        hea_name = Path(hea_file.filename).name
                        dat_name = Path(dat_file.filename).name
                        hea_path = Path(tmp_dir) / hea_name
                        dat_path = Path(tmp_dir) / dat_name

                        hea_file.save(str(hea_path))
                        dat_file.save(str(dat_path))

                        record_path = str(hea_path.with_suffix(''))
                        signal_matrix, _ = wfdb.rdsamp(record_path)
                        signal_matrix = signal_matrix[:, :12]

                        result = analyze_signal_cnn(signal_matrix, threshold=threshold)
                        signal = signal_matrix[:, 1].astype(float).tolist() if signal_matrix.shape[1] > 1 else signal_matrix[:, 0].astype(float).tolist()

                        rec = {
                            'name': f"{hea_name} + {dat_name}",
                            'signal': signal,
                            'signal_matrix': signal_matrix.astype(float).tolist(),
                            'source_type': 'wfdb_upload',
                            'patient_id': patient_data['patient_id'],
                            'patient_name': patient_data['name'],
                            'patient_gender': patient_data['gender'],
                            'patient_age': patient_data['age'],
                            'patient_phone': patient_data['phone'],
                            'patient_address': patient_data['address'],
                            'patient_notes': patient_data['notes'],
                            'patient': patient_data,
                            'timestamp': datetime.datetime.utcnow(),
                            **result,
                        }
                        res = ecgs_collection.insert_one(rec)
                        _insert_ecg_record(res.inserted_id, rec, patient_data)
                        return redirect(url_for('view_ecg', ecg_id=res.inserted_id))
                except Exception as e:
                    return f"Loi xu ly WFDB: {e}", 400

        # Default CSV upload mode.
        f = request.files.get('file')
        if f:
            name = f.filename
            t1 = time.time()
            print(f"read_csv took {t1-t0:.3f}s")
            extracted = _extract_signal_from_uploaded_csv(f)
            signal = extracted['signal']
            signal_matrix = extracted['signal_matrix']
            result = _analyze_uploaded_payload(signal, signal_matrix=signal_matrix, threshold=threshold)

            # ensure result values are serializable and friendly
            if 'heart_rate' in result:
                if result['heart_rate'] is not None:
                    result['heart_rate'] = float(result['heart_rate'])
                else:
                    result['heart_rate'] = None
            if 'prediction' in result and isinstance(result['prediction'], np.generic):
                result['prediction'] = result['prediction'].item()
            if result.get('model_type') == 'random_forest_legacy' and 'prediction' in result and result['prediction'] is not None:
                result['prediction'] = int(result['prediction'])
            rec = {'name': name, 'signal': signal,
                   'patient_id': patient_data['patient_id'],
                   'patient_name': patient_data['name'],
                   'patient_gender': patient_data['gender'],
                   'patient_age': patient_data['age'],
                   'patient_phone': patient_data['phone'],
                   'patient_address': patient_data['address'],
                   'patient_notes': patient_data['notes'],
                   'patient': patient_data,
                   'timestamp': datetime.datetime.utcnow(),
                   **result}
            if signal_matrix is not None:
                rec['signal_matrix'] = signal_matrix
            if extracted.get('source_type'):
                rec['source_type'] = extracted['source_type']
            if extracted.get('source_ref'):
                rec['source_ref'] = extracted['source_ref']
            res = ecgs_collection.insert_one(rec)
            _insert_ecg_record(res.inserted_id, rec, patient_data)
            t2 = time.time()
            print(f"mongo insert took {t2-t1:.3f}s, total {t2-t0:.3f}s")
            return redirect(url_for('view_ecg', ecg_id=res.inserted_id))
    # --- Sinh mã bệnh nhân tự động tăng ---
    # Tìm mã lớn nhất dạng BNxxx
    last_patient = patients_collection.find_one(
        {'patient_id': {'$regex': r'^BN\\d+$'}},
        sort=[('patient_id', -1)]
    )
    next_patient_id = "BN001"
    if last_patient and 'patient_id' in last_patient:
        import re
        m = re.match(r'^BN(\d+)$', last_patient['patient_id'])
        if m:
            next_num = int(m.group(1)) + 1
            next_patient_id = f"BN{next_num:03d}"

    # Lấy mã BN gần nhất
    last_patient = patients_collection.find_one(
        {'patient_id': {'$regex': r'^BN\d+$'}},
        sort=[('patient_id', -1)]
    )
    last_patient_id = last_patient['patient_id'] if last_patient and 'patient_id' in last_patient else None

    return render_template(
        'upload.html',
        threshold_defaults_json=json.dumps(get_active_thresholds(), ensure_ascii=False),
        next_patient_id=next_patient_id,
        last_patient_id=last_patient_id,
    )


def _parse_patient_form(form):
    patient_id = (form.get('patient_id') or '').strip()
    name = (form.get('patient_name') or '').strip()
    gender = (form.get('patient_gender') or '').strip()
    age_raw = (form.get('patient_age') or '').strip()
    phone = (form.get('patient_phone') or '').strip()
    address = (form.get('patient_address') or '').strip()
    notes = (form.get('patient_notes') or '').strip()

    if not patient_id:
        return None, 'Vui lòng nhập mã bệnh nhân.'
    if not name:
        return None, 'Vui lòng nhập họ và tên bệnh nhân.'
    if gender not in ('Nam', 'Nữ'):
        return None, 'Vui lòng chọn giới tính hợp lệ (Nam/Nữ).'

    age = None
    if age_raw:
        try:
            age = int(age_raw)
            if age <= 0 or age > 130:
                raise ValueError
        except Exception:
            return None, 'Tuổi không hợp lệ. Vui lòng nhập số từ 1 đến 130.'

    return {
        'patient_id': patient_id,
        'name': name,
        'gender': gender,
        'age': age,
        'phone': phone,
        'address': address,
        'notes': notes,
    }, None


def _ensure_patient_exists(patient_data):
    existing = patients_collection.find_one({'patient_id': patient_data['patient_id']})
    if existing:
        return existing

    doc = {
        'patient_id': patient_data['patient_id'],
        'name': patient_data.get('name'),
        'gender': patient_data.get('gender'),
        'age': patient_data.get('age'),
        'phone': patient_data.get('phone'),
        'address': patient_data.get('address'),
        'notes': patient_data.get('notes'),
        'created_at': datetime.datetime.utcnow(),
    }
    patients_collection.insert_one(doc)
    return doc


def _insert_ecg_record(ecg_id, ecg_doc, patient_data):
    predictions = {
        'prediction': ecg_doc.get('prediction'),
        'prediction_label': ecg_doc.get('prediction_label'),
        'prediction_confidence': ecg_doc.get('prediction_confidence'),
        'multi_label_predictions': ecg_doc.get('multi_label_predictions', []),
    }
    record_doc = {
        'ecg_id': ecg_id,
        'patient_id': patient_data['patient_id'],
        'patient_name': patient_data.get('name'),
        'gender': patient_data.get('gender'),
        'age': patient_data.get('age'),
        'phone': patient_data.get('phone'),
        'address': patient_data.get('address'),
        'notes': patient_data.get('notes'),
        'ecg_file': ecg_doc.get('name'),
        'bpm': ecg_doc.get('heart_rate'),
        'predictions': predictions,
        'prediction': ecg_doc.get('prediction'),
        'prediction_label': ecg_doc.get('prediction_label'),
        'prediction_confidence': ecg_doc.get('prediction_confidence'),
        'multi_label_predictions': ecg_doc.get('multi_label_predictions', []),
        'created_at': ecg_doc.get('timestamp', datetime.datetime.utcnow()),
    }
    ecg_records_collection.insert_one(record_doc)


def _record_time(doc):
    return doc.get('created_at') or doc.get('timestamp')


def _record_predictions(doc):
    nested = doc.get('predictions') if isinstance(doc.get('predictions'), dict) else {}
    multi = doc.get('multi_label_predictions') or nested.get('multi_label_predictions') or []
    prediction = doc.get('prediction', nested.get('prediction'))
    prediction_label = doc.get('prediction_label', nested.get('prediction_label'))
    prediction_confidence = doc.get('prediction_confidence', nested.get('prediction_confidence'))
    prediction_desc = doc.get('prediction_desc', nested.get('prediction_desc'))

    # Backfill description for legacy/new records where prediction_desc is missing.
    if not prediction_desc:
        code_to_desc = {v: CLASS_LABELS.get(k, {}).get('desc') for k, v in LEGACY_CLASS_CODES.items()}

        # Try by explicit prediction label/code first.
        if prediction_label:
            prediction_desc = code_to_desc.get(str(prediction_label).upper())

        # Try by numeric class id.
        if not prediction_desc and prediction is not None:
            try:
                prediction_desc = CLASS_LABELS.get(int(prediction), {}).get('desc')
            except Exception:
                prediction_desc = None

        # Fallback to top multi-label disease code.
        if not prediction_desc and multi:
            top = multi[0] if isinstance(multi[0], dict) else None
            if top:
                prediction_desc = code_to_desc.get(str(top.get('disease', '')).upper())

    return {
        'multi_label_predictions': multi,
        'prediction': prediction,
        'prediction_label': prediction_label,
        'prediction_confidence': prediction_confidence,
        'prediction_desc': prediction_desc,
    }


def _record_patient(doc):
    patient = doc.get('patient') if isinstance(doc.get('patient'), dict) else {}
    patient_id = doc.get('patient_id', patient.get('patient_id'))
    patient_name = doc.get('patient_name', patient.get('name'))
    gender = doc.get('gender', patient.get('gender'))
    age = doc.get('age', patient.get('age'))
    phone = doc.get('phone', patient.get('phone'))
    address = doc.get('address', patient.get('address'))
    notes = doc.get('notes', patient.get('notes'))

    # Backfill from patients collection for legacy records that miss patient_id.
    if (not patient_id) and (patient_name or phone):
        try:
            ors = []
            if phone:
                ors.append({'phone': phone})
            if patient_name:
                ors.append({'name': patient_name})
            if ors:
                pdoc = patients_collection.find_one({'$or': ors}, {'patient_id': 1, 'name': 1, 'gender': 1, 'age': 1, 'phone': 1, 'address': 1, 'notes': 1})
                if pdoc:
                    patient_id = patient_id or pdoc.get('patient_id')
                    patient_name = patient_name or pdoc.get('name')
                    gender = gender or pdoc.get('gender')
                    age = age if age is not None else pdoc.get('age')
                    phone = phone or pdoc.get('phone')
                    address = address or pdoc.get('address')
                    notes = notes or pdoc.get('notes')
        except Exception:
            pass

    return {
        'patient_id': patient_id,
        'patient_name': patient_name,
        'gender': gender,
        'age': age,
        'phone': phone,
        'address': address,
        'notes': notes,
    }


def _resolve_signal_from_record(record_doc):
    ecg_ref = record_doc.get('ecg_id')
    if ecg_ref is None:
        return None

    oid = None
    if isinstance(ecg_ref, ObjectId):
        oid = ecg_ref
    else:
        try:
            oid = ObjectId(str(ecg_ref))
        except Exception:
            oid = None

    if oid is None:
        return None
    src = ecgs_collection.find_one({'_id': oid}, {'signal': 1})
    if not src:
        return None
    return src.get('signal')

@app.route('/ecgs')
def list_ecgs():
    ecgs = list(ecgs_collection.find({}, {'signal':0}))
    return render_template('list.html', ecgs=ecgs)

@app.route('/history')
def history():
    q = request.args.get('q', '').strip()
    period = request.args.get('period', '7')  # '7' days, '30' days, '12m' or ignored when custom
    start_date_str = request.args.get('start_date', '').strip()
    end_date_str = request.args.get('end_date', '').strip()
    page = int(request.args.get('page', 1))
    per_page = 10
    query = {}
    if q:
        ors = []
        text_fields = [
            'ecg_file',
            'patient_name',
            'patient_id',
            'patient.gender',
            'patient.age',
            'patient_phone',
            'patient_address',
            'patient_notes',
            'name',
            'gender',
            'age',
            'phone',
            'address',
            'notes',
        ]
        try:
            age_q = int(q)
            ors.append({'age': age_q})
            ors.append({'patient.age': age_q})
        except Exception:
            pass
        # try treat query as ObjectId
        try:
            oid = ObjectId(q)
            ors.append({'_id': oid})
            ors.append({'ecg_id': oid})
        except Exception:
            pass
        # try parse as date (dd/mm/YYYY or ISO-like)
        parsed = None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.datetime.strptime(q, fmt)
                break
            except Exception:
                parsed = None
        if parsed:
            # build a day-range (naive datetimes to match stored values)
            start = datetime.datetime(parsed.year, parsed.month, parsed.day, 0, 0, 0)
            end = start + datetime.timedelta(days=1)
            ors.append({'created_at': {'$gte': start, '$lt': end}})
        # final query is OR of possible matches
        if ors:
            query = {'$or': ors}
    # parse custom date range early so it can affect the query
    today = datetime.datetime.utcnow().date()
    labels = []
    values = []
    custom_start = None
    custom_end = None
    def parse_date(s):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.datetime.strptime(s, fmt).date()
            except Exception:
                continue
        return None
    if start_date_str and end_date_str:
        sd = parse_date(start_date_str)
        ed = parse_date(end_date_str)
        if sd and ed and sd <= ed:
            custom_start = sd
            custom_end = ed
            # when custom range specified, ignore period radios
            period = ''
    if custom_start and custom_end:
        start_dt = datetime.datetime(custom_start.year, custom_start.month, custom_start.day)
        end_dt = datetime.datetime(custom_end.year, custom_end.month, custom_end.day) + datetime.timedelta(days=1)
        query["created_at"] = {"$gte": start_dt, "$lt": end_dt}
    # now perform count/find with full query
    total = ecg_records_collection.count_documents(query)
    docs = list(ecg_records_collection.find(query).sort('created_at', -1)
                .skip((page-1)*per_page).limit(per_page))
    ecgs = []
    for d in docs:
        p = _record_patient(d)
        preds = _record_predictions(d)
        ecgs.append({
            '_id': d.get('_id'),
            'view_id': d.get('ecg_id') or d.get('_id'),
            'name': d.get('ecg_file') or d.get('name') or '--',
            'timestamp': _record_time(d),
            'heart_rate': d.get('bpm', d.get('heart_rate')),
            'patient_name': p.get('patient_name'),
            'age': p.get('age'),
            'gender': p.get('gender'),
            **preds,
        })
    # build chart labels/values based on either custom or period
    if custom_start and custom_end:
        cur = custom_start
        while cur <= custom_end:
            nextday = cur + datetime.timedelta(days=1)
            start = datetime.datetime(cur.year, cur.month, cur.day)
            end = datetime.datetime(nextday.year, nextday.month, nextday.day)
            num = ecg_records_collection.count_documents({'created_at': {'$gte': start, '$lt': end}})
            labels.append(cur.strftime('%d/%m/%Y'))
            values.append(num)
            cur = nextday
    else:
        if period == '7':
            for i in range(6, -1, -1):
                day = today - datetime.timedelta(days=i)
                start = datetime.datetime(day.year, day.month, day.day)
                end = start + datetime.timedelta(days=1)
                num = ecg_records_collection.count_documents({'created_at': {'$gte': start, '$lt': end}})
                labels.append(day.strftime('%d/%m'))
                values.append(num)
        elif period == '30':
            for i in range(29, -1, -1):
                day = today - datetime.timedelta(days=i)
                start = datetime.datetime(day.year, day.month, day.day)
                end = start + datetime.timedelta(days=1)
                num = ecg_records_collection.count_documents({'created_at': {'$gte': start, '$lt': end}})
                labels.append(day.strftime('%d/%m'))
                values.append(num)
        elif period == '12m':
            def subtract_months(date, n):
                year = date.year
                month = date.month - n
                while month <= 0:
                    month += 12
                    year -= 1
                return datetime.date(year, month, 1)
            start_month = today.replace(day=1)
            for i in range(11, -1, -1):
                mdate = subtract_months(start_month, i)
                start = datetime.datetime(mdate.year, mdate.month, mdate.day)
                if mdate.month == 12:
                    end = datetime.datetime(mdate.year + 1, 1, 1)
                else:
                    end = datetime.datetime(mdate.year, mdate.month + 1, 1)
                num = ecg_records_collection.count_documents({'created_at': {'$gte': start, '$lt': end}})
                labels.append(mdate.strftime('%m/%Y'))
                values.append(num)
        else:
            for i in range(6, -1, -1):
                day = today - datetime.timedelta(days=i)
                start = datetime.datetime(day.year, day.month, day.day)
                end = start + datetime.timedelta(days=1)
                num = ecg_records_collection.count_documents({'created_at': {'$gte': start, '$lt': end}})
                labels.append(day.strftime('%d/%m'))
                values.append(num)
    total_pages = (total + per_page - 1)//per_page
    return render_template('history.html', ecgs=ecgs, chart_labels=labels, chart_values=values,
                           page=page, total_pages=total_pages, q=q, period=period)


@app.route('/history/suggest')
def history_suggest():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    suggestions = []
    # suggest matching patient names and file names first
    try:
        names = ecg_records_collection.distinct('patient_name', {'patient_name': {'$regex': q, '$options': 'i'}})
        for n in names:
            if n and n not in suggestions:
                suggestions.append(n)
        nested_names = ecg_records_collection.distinct('patient.name', {'patient.name': {'$regex': q, '$options': 'i'}})
        for n in nested_names:
            if n and n not in suggestions:
                suggestions.append(n)
        patient_ids = ecg_records_collection.distinct('patient_id', {'patient_id': {'$regex': q, '$options': 'i'}})
        for pid in patient_ids:
            if pid and pid not in suggestions:
                suggestions.append(pid)
        phones = ecg_records_collection.distinct('phone', {'phone': {'$regex': q, '$options': 'i'}})
        for ph in phones:
            if ph and ph not in suggestions:
                suggestions.append(ph)
        files = ecg_records_collection.distinct('ecg_file', {'ecg_file': {'$regex': q, '$options': 'i'}})
        for f in files:
            if f and f not in suggestions:
                suggestions.append(f)
    except Exception:
        pass
    # also include any ObjectId hex strings that contain the query (scan recent docs)
    try:
        if len(suggestions) < 20:
            for d in ecg_records_collection.find({}, {'_id': 1}).sort('created_at', -1).limit(200):
                sid = str(d['_id'])
                if q in sid and sid not in suggestions:
                    suggestions.append(sid)
                    if len(suggestions) >= 20:
                        break
    except Exception:
        pass
    return jsonify(suggestions[:20])

@app.route('/ecg/<ecg_id>')
def view_ecg(ecg_id):
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return 'ID không hợp lệ', 400
    rec = ecgs_collection.find_one({'_id': oid})
    if not rec:
        return 'Không tìm thấy', 404
    return render_template('view.html', ecg=rec, disease_details=DISEASE_DETAILS)


@app.route('/ecg/<ecg_id>/print')
def print_ecg(ecg_id):
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return 'ID không hợp lệ', 400
    rec = ecgs_collection.find_one({'_id': oid})
    if not rec:
        return 'Không tìm thấy', 404
    # render minimal template for printing the ECG chart only
    return render_template('print_ecg.html', ecg=rec, disease_details=DISEASE_DETAILS)

@app.route('/delete/<ecg_id>')
def delete_ecg(ecg_id):
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return 'ID không hợp lệ', 400

    # History is sourced from ecg_records; remove matching records first.
    matched_records = list(ecg_records_collection.find(
        {'$or': [{'_id': oid}, {'ecg_id': oid}]},
        {'_id': 1, 'ecg_id': 1}
    ))

    if matched_records:
        ecg_records_collection.delete_many({'$or': [{'_id': oid}, {'ecg_id': oid}]})

    # Also remove linked signal docs in legacy ecgs collection.
    linked_ecg_ids = set()
    linked_ecg_ids.add(oid)
    for rec in matched_records:
        linked = rec.get('ecg_id')
        if isinstance(linked, ObjectId):
            linked_ecg_ids.add(linked)

    if linked_ecg_ids:
        ecgs_collection.delete_many({'_id': {'$in': list(linked_ecg_ids)}})

    return redirect(url_for('history'))

@app.route('/export/<ecg_id>')
def export_ecg(ecg_id):
    # PDF export disabled on Windows or missing libraries
    return "Chức năng xuất PDF chưa khả dụng", 501

@app.route('/history/export')
def export_history():
    q = request.args.get('q', '').strip()
    period = request.args.get('period', '7')
    start_date_str = request.args.get('start_date', '').strip()
    end_date_str = request.args.get('end_date', '').strip()
    query = {}
    if q:
        ors = []
        text_fields = [
            'patient_name', 'patient_id', 'phone', 'ecg_file',
            'patient.name', 'patient.patient_id', 'patient.phone',
        ]
        for field in text_fields:
            ors.append({field: {'$regex': q, '$options': 'i'}})
        try:
            age_q = int(q)
            ors.append({'age': age_q})
            ors.append({'patient.age': age_q})
        except Exception:
            pass
        query['$or'] = ors
    today = datetime.datetime.utcnow().date()
    # parse custom
    def parse_date(s):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
            try:
                return datetime.datetime.strptime(s, fmt).date()
            except Exception:
                continue
        return None
    if start_date_str and end_date_str:
        sd = parse_date(start_date_str)
        ed = parse_date(end_date_str)
        if sd and ed and sd <= ed:
            start = datetime.datetime(sd.year, sd.month, sd.day)
            end = datetime.datetime(ed.year, ed.month, ed.day) + datetime.timedelta(days=1)
            query['created_at'] = {'$gte': start, '$lt': end}
    else:
        # apply same time bounds as history chart
        if period == '7':
            start = datetime.datetime(today.year, today.month, today.day) - datetime.timedelta(days=6)
            query['created_at'] = {'$gte': start}
        elif period == '30':
            start = datetime.datetime(today.year, today.month, today.day) - datetime.timedelta(days=29)
            query['created_at'] = {'$gte': start}
        elif period == '12m':
            year = today.year
            month = today.month
            month -= 11
            while month <= 0:
                month += 12
                year -= 1
            start = datetime.datetime(year, month, 1)
            query['created_at'] = {'$gte': start}
    docs = list(ecg_records_collection.find(query).sort('created_at', -1))
    # generate csv
    import csv
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['_id','patient_name','age','gender','ecg_file','created_at','bpm','prediction'])
    for d in docs:
        preds = _record_predictions(d)
        cw.writerow([
            str(d.get('_id')),
            d.get('patient_name', ''),
            d.get('age', ''),
            d.get('gender', ''),
            d.get('ecg_file', ''),
            _record_time(d),
            d.get('bpm', ''),
            preds.get('prediction', ''),
        ])
    output = si.getvalue()
    return output, 200, {'Content-Type':'text/csv', 'Content-Disposition':'attachment; filename="history.csv"'}


@app.route('/settings/thresholds', methods=['GET', 'POST'])
def threshold_settings():
    from flask import flash

    if request.method == 'POST':
        raw_json = (request.form.get('thresholds_json') or '').strip()
        parsed = None
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except Exception:
                flash('Threshold JSON không hợp lệ.', 'danger')
                return redirect(url_for('threshold_settings'))
        else:
            parsed = {}
            for disease in DEFAULT_LABEL_THRESHOLDS.keys():
                val = (request.form.get(f'th_{disease}') or '').strip()
                if val == '':
                    continue
                try:
                    parsed[disease] = float(val)
                except Exception:
                    flash(f'Giá trị không hợp lệ cho {disease}.', 'danger')
                    return redirect(url_for('threshold_settings'))

        save_active_thresholds(parsed)
        flash('Đã cập nhật threshold cho mô hình CNN.', 'success')
        return redirect(url_for('threshold_settings'))

    active = get_active_thresholds()
    return render_template('threshold_settings.html', thresholds=active)

# --- API -----------------------------------------------------------
@app.route('/analyze', methods=['GET'])
def analyze_home():
    """New unified ECG analysis page (upload + waveform + AI results)."""
    return render_template(
        'analyze_ecg.html',
        threshold_defaults_json=json.dumps(get_active_thresholds(), ensure_ascii=False),
    )


@app.route('/model-evaluation')
def model_evaluation():
    """Display CNN model evaluation metrics and training history."""
    class_report_raw = {}
    training_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    report_path = ROOT_DIR / 'ml' / 'cnn_model' / 'classification_report.json'
    history_path = ROOT_DIR / 'ml' / 'cnn_model' / 'training_history.json'

    if report_path.exists():
        try:
            with open(report_path, encoding='utf-8') as f:
                class_report_raw = json.load(f)
        except Exception:
            pass

    if history_path.exists():
        try:
            with open(history_path, encoding='utf-8') as f:
                training_history = json.load(f)
        except Exception:
            pass

    # Normalize keys: 'f1-score' -> 'f1_score', 'micro avg' -> 'micro_avg', etc.
    key_remap = {'micro avg': 'micro_avg', 'macro avg': 'macro_avg',
                 'weighted avg': 'weighted_avg', 'samples avg': 'samples_avg'}
    class_report = {}
    for k, v in class_report_raw.items():
        nk = key_remap.get(k, k)
        if isinstance(v, dict):
            class_report[nk] = {vk.replace('-', '_').replace(' ', '_'): vv for vk, vv in v.items()}
        else:
            class_report[nk] = v

    return render_template(
        'model_evaluation.html',
        class_report=class_report,
        training_history=training_history,
        disease_vn=DISEASE_VN,
    )


@app.route('/report')
def report_page():
    """Report generation page with recent ECG list."""
    recent_docs = list(ecg_records_collection.find({}).sort('created_at', -1).limit(20))
    recent_ecgs = []
    for d in recent_docs:
        preds = _record_predictions(d)
        recent_ecgs.append({
            '_id': d.get('_id'),
            'name': d.get('ecg_file') or d.get('name') or '--',
            'timestamp': _record_time(d),
            'multi_label_predictions': preds.get('multi_label_predictions') or [],
            'prediction_label': preds.get('prediction_label'),
            'patient_name': d.get('patient_name'),
        })
    return render_template('report.html', recent_ecgs=recent_ecgs, disease_vn=DISEASE_VN)


@app.route('/api/ecg/<ecg_id>')
def api_get_ecg(ecg_id):
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return jsonify({'error': 'invalid id'}), 400
    rec = ecgs_collection.find_one({'_id': oid})
    if rec and rec.get('signal') is not None:
        return jsonify({'signal': rec['signal']})

    record = ecg_records_collection.find_one({'_id': oid})
    if record:
        signal = _resolve_signal_from_record(record)
        if signal is not None:
            return jsonify({'signal': signal})

    return jsonify({'error': 'not found'}), 404


@app.route('/api/ecg-detail/<ecg_id>')
def api_ecg_detail(ecg_id):
    """Return full ECG record (without large signal_matrix) for report page."""
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return jsonify({'error': 'invalid id'}), 400
    rec = ecg_records_collection.find_one({'_id': oid})
    if not rec:
        return jsonify({'error': 'not found'}), 404

    preds = _record_predictions(rec)
    patient = _record_patient(rec)
    signal = _resolve_signal_from_record(rec) or []
    result = {
        '_id': str(rec.get('_id')),
        'ecg_id': str(rec.get('ecg_id')) if rec.get('ecg_id') is not None else None,
        'name': rec.get('ecg_file') or rec.get('name') or '--',
        'timestamp': str(_record_time(rec)) if _record_time(rec) is not None else '',
        'timestamp_vn': format_datetime(_record_time(rec)) if _record_time(rec) is not None else '',
        'heart_rate': rec.get('bpm', rec.get('heart_rate')),
        'prediction': preds.get('prediction'),
        'prediction_label': preds.get('prediction_label'),
        'prediction_confidence': preds.get('prediction_confidence'),
        'prediction_desc': preds.get('prediction_desc'),
        'multi_label_predictions': preds.get('multi_label_predictions') or [],
        'patient_id': patient.get('patient_id'),
        'patient_name': patient.get('patient_name'),
        'patient_age': patient.get('age'),
        'patient_gender': patient.get('gender'),
        'phone': patient.get('phone'),
        'address': patient.get('address'),
        'notes': patient.get('notes'),
        'patient': {
            'patient_id': patient.get('patient_id'),
            'name': patient.get('patient_name'),
            'age': patient.get('age'),
            'gender': patient.get('gender'),
            'phone': patient.get('phone'),
            'address': patient.get('address'),
            'notes': patient.get('notes'),
        },
        'signal': signal,
    }
    for p in result.get('multi_label_predictions', []):
        p['disease_vn'] = DISEASE_VN.get(p.get('disease', ''), p.get('disease', ''))
    # Convert numpy types
    def _safe(v):
        if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')):
            return None
        return v
    if result.get('heart_rate') is not None:
        result['heart_rate'] = _safe(float(result['heart_rate']))
    return jsonify(result)


@app.route('/print-ecg/<ecg_id>')
def print_ecg_page(ecg_id):
    """Render print-friendly ECG view."""
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return 'ID kh\u00f4ng h\u1ee3p l\u1ec7', 400
    rec = ecg_records_collection.find_one({'_id': oid})
    if not rec:
        return 'Kh\u00f4ng t\u00ecm th\u1ea5y', 404

    preds = _record_predictions(rec)
    patient = _record_patient(rec)
    view_doc = {
        '_id': rec.get('_id'),
        'ecg_id': rec.get('ecg_id'),
        'name': rec.get('ecg_file') or rec.get('name') or '--',
        'timestamp': _record_time(rec),
        'heart_rate': rec.get('bpm', rec.get('heart_rate')),
        'prediction': preds.get('prediction'),
        'prediction_label': preds.get('prediction_label'),
        'prediction_confidence': preds.get('prediction_confidence'),
        'prediction_desc': preds.get('prediction_desc'),
        'multi_label_predictions': preds.get('multi_label_predictions') or [],
        'patient_id': patient.get('patient_id'),
        'patient_name': patient.get('patient_name'),
        'patient_age': patient.get('age'),
        'patient_gender': patient.get('gender'),
        'patient_phone': patient.get('phone'),
        'patient_address': patient.get('address'),
        'patient_notes': patient.get('notes'),
        'patient': {
            'patient_id': patient.get('patient_id'),
            'name': patient.get('patient_name'),
            'age': patient.get('age'),
            'gender': patient.get('gender'),
            'phone': patient.get('phone'),
            'address': patient.get('address'),
            'notes': patient.get('notes'),
        },
    }
    return render_template('print_ecg.html', ecg=view_doc, disease_details=DISEASE_DETAILS)

@app.route('/api/analyze/<ecg_id>', methods=['POST'])
def api_analyze(ecg_id):
    # existing JSON endpoint (used for AJAX or external callers)
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return jsonify({'error': 'invalid id'}), 400
    rec = ecgs_collection.find_one({'_id': oid})
    if not rec:
        return jsonify({'error':'not found'}), 404
    if rec.get('signal_matrix') is not None and cnn_predict_ecg_array is not None:
        result = analyze_signal_cnn(rec['signal_matrix'])
    else:
        result = analyze_signal(rec['signal'])
    ecgs_collection.update_one({'_id': oid}, {'$set': result})
    return jsonify(result)

@app.route('/analyze/<ecg_id>', methods=['POST'])
def analyze(ecg_id):
    # form submission from UI; perform analysis and redirect back
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return 'ID không hợp lệ', 400
    rec = ecgs_collection.find_one({'_id': oid})
    if not rec:
        return 'Không tìm thấy', 404
    if rec.get('signal_matrix') is not None and cnn_predict_ecg_array is not None:
        result = analyze_signal_cnn(rec['signal_matrix'])
    else:
        result = analyze_signal(rec['signal'])
    ecgs_collection.update_one({'_id': oid}, {'$set': result})
    # notify user
    from flask import flash
    flash('Đã phân tích lại ECG. Kết quả mới đã cập nhật.', 'success')
    return redirect(url_for('view_ecg', ecg_id=ecg_id))

# legacy prediction (for frontend upload)
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    signal = data.get('signal', [])
    threshold = data.get('threshold')
    threshold_map = data.get('threshold_map')
    mode = str(data.get('model_type', 'auto')).lower()

    if threshold_map is not None:
        threshold = threshold_map
    if threshold is None:
        threshold = get_active_thresholds()

    if mode in ('auto', 'cnn') and cnn_predict_ecg_array is not None:
        try:
            out = analyze_signal_cnn(signal, threshold=threshold)
            return jsonify(out)
        except Exception as e:
            if mode == 'cnn':
                return jsonify({'error': f'cnn inference failed: {e}'}), 400

    out = analyze_signal(signal)
    return jsonify(out)


@app.route('/api/predict-file', methods=['POST'])
def predict_file():
    if cnn_predict_ecg is None or cnn_model is None or cnn_labels is None:
        return jsonify({'error': 'cnn model is not available'}), 503

    data = request.json or {}
    file_path = data.get('file_path')
    threshold = data.get('threshold')
    threshold_map = data.get('threshold_map')
    if threshold_map is not None:
        threshold = threshold_map
    if threshold is None:
        threshold = get_active_thresholds()
    if not file_path:
        return jsonify({'error': 'file_path is required'}), 400

    try:
        preds = cnn_predict_ecg(
            file_path=file_path,
            threshold=threshold,
            model=cnn_model,
            label_names=cnn_labels,
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({
        'model_type': 'cnn_ptbxl_20',
        'inference_meta': {
            'model_type': 'cnn_ptbxl_20',
            'model_version': get_cnn_model_version(),
            'inference_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'threshold_applied': _resolve_thresholds(threshold),
        },
        'file_path': file_path,
        'threshold': threshold,
        'n_predictions': len(preds),
        'multi_label_predictions': preds,
    })


@app.route('/api/predict-upload', methods=['POST'])
def predict_upload():
    threshold_raw = request.form.get('threshold', '0.5')
    threshold = threshold_raw if threshold_raw != '' else None

    threshold_map_raw = request.form.get('threshold_map')
    threshold_map = None
    if threshold_map_raw:
        try:
            threshold_map = json.loads(threshold_map_raw)
        except Exception:
            threshold_map = None

    resolved_threshold = threshold_map if threshold_map is not None else threshold
    if resolved_threshold is None:
        resolved_threshold = get_active_thresholds()

    csv_file = request.files.get('file')
    hea_file = request.files.get('hea_file')
    dat_file = request.files.get('dat_file')

    if csv_file:
        try:
            extracted = _extract_signal_from_uploaded_csv(csv_file)
            out = _analyze_uploaded_payload(
                extracted['signal'],
                signal_matrix=extracted['signal_matrix'],
                threshold=resolved_threshold,
            )
            if extracted.get('source_ref'):
                out.setdefault('inference_meta', {})['source_ref'] = extracted['source_ref']
            return jsonify(out)
        except Exception as e:
            return jsonify({'error': f'csv inference failed: {e}'}), 400

    if hea_file and dat_file:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                hea_name = Path(hea_file.filename).name
                dat_name = Path(dat_file.filename).name

                hea_path = Path(tmp_dir) / hea_name
                dat_path = Path(tmp_dir) / dat_name

                hea_file.save(str(hea_path))
                dat_file.save(str(dat_path))

                record_path = str(hea_path.with_suffix(''))
                signal_matrix, _ = wfdb.rdsamp(record_path)
                signal_matrix = np.asarray(signal_matrix[:, :12], dtype=float)
                chart_signal = signal_matrix[:, 1] if signal_matrix.shape[1] > 1 else signal_matrix[:, 0]
                out = _analyze_uploaded_payload(
                    chart_signal.astype(float).tolist(),
                    signal_matrix=signal_matrix.astype(float).tolist(),
                    threshold=resolved_threshold,
                )
                out.setdefault('inference_meta', {})['source_ref'] = record_path
                return jsonify(out)
        except Exception as e:
            return jsonify({'error': f'wfdb upload inference failed: {e}'}), 400

    return jsonify({'error': 'provide either file (CSV) or both hea_file and dat_file'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
