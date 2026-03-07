from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import os
import io
import datetime
import numpy as np
import joblib
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
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

# load model if available
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

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
    if model is not None:
        arr = np.array(proc)
        # align to expected feature count if possible
        if hasattr(model, 'n_features_in_'):
            exp = model.n_features_in_
            if arr.size != exp:
                if arr.size == exp + 1:
                    arr = arr[:-1]
                elif arr.size == exp - 1:
                    arr = np.pad(arr, (0, exp - arr.size))
                else:
                    print(f"predict skipped: got {arr.size} features, expected {exp}")
                    return {'heart_rate': float(hr), 'prediction': None}
        try:
            pred = model.predict([arr.tolist()])[0]
        except Exception as e:
            print("prediction error", e)
            pred = None
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
    return result

# --- web routes -----------------------------------------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    count = ecgs_collection.count_documents({})
    # compute average heart rate if available
    hr_docs = ecgs_collection.find({'heart_rate': {'$exists': True}}, {'heart_rate': 1})
    hr_list = [d['heart_rate'] for d in hr_docs]
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
    ai_status = 'Online' if model is not None else 'Offline'
    return render_template('dashboard.html', count=count, avg_hr=avg_hr, chart_labels=labels, chart_values=values, ai_status=ai_status)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        import time
        t0 = time.time()
        f = request.files.get('file')
        if f:
            name = f.filename
            df = pd.read_csv(f, header=None)
            t1 = time.time()
            print(f"read_csv took {t1-t0:.3f}s")
            # flatten everything; if last column appears to be a label integer
            # (e.g. MIT‑BIH format), drop it before storing.
            arr = df.values.astype(float)
            flat = arr.flatten()
            # drop label value if it appears at the end (MIT‑BIH format)
            if flat.size > 1 and float(flat[-1]).is_integer() and flat[-1] in (0,1,2,3,4):
                flat = flat[:-1]
            signal = flat.tolist()
            # if we have a trained model, try to reconcile feature count
            if model is not None and hasattr(model, 'n_features_in_'):
                exp = model.n_features_in_
                if len(signal) == exp + 1:
                    # maybe label was at the beginning instead of end
                    if isinstance(signal[0], (int, float)) and float(signal[0]).is_integer() and signal[0] in (0,1,2,3,4):
                        signal = signal[1:]
                    else:
                        signal = signal[:-1]
                elif len(signal) == exp - 1:
                    # one missing value; pad with zero
                    signal = signal + [0.0]
                # if lengths still mismatched we leave it and let analyze_signal warn
            # immediately analyze so the detail page can show results
            result = analyze_signal(signal)
            # ensure result values are serializable and friendly
            if 'heart_rate' in result:
                if result['heart_rate'] is not None:
                    result['heart_rate'] = float(result['heart_rate'])
                else:
                    result['heart_rate'] = None
            if 'prediction' in result and result['prediction'] is not None:
                result['prediction'] = int(result['prediction'])
            rec = {'name': name, 'signal': signal,
                   'timestamp': datetime.datetime.utcnow(),
                   **result}
            res = ecgs_collection.insert_one(rec)
            t2 = time.time()
            print(f"mongo insert took {t2-t1:.3f}s, total {t2-t0:.3f}s")
            return redirect(url_for('view_ecg', ecg_id=res.inserted_id))
    return render_template('upload.html')

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
        # search by name (case-insensitive substring)
        ors.append({'name': {'$regex': q, '$options': 'i'}})
        # try treat query as ObjectId
        try:
            oid = ObjectId(q)
            ors.append({'_id': oid})
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
            ors.append({'timestamp': {'$gte': start, '$lt': end}})
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
        query["timestamp"] = {"$gte": start_dt, "$lt": end_dt}
    # now perform count/find with full query
    total = ecgs_collection.count_documents(query)
    ecgs = list(ecgs_collection.find(query).sort('timestamp', -1)
                .skip((page-1)*per_page).limit(per_page))
    # build chart labels/values based on either custom or period
    if custom_start and custom_end:
        cur = custom_start
        while cur <= custom_end:
            nextday = cur + datetime.timedelta(days=1)
            start = datetime.datetime(cur.year, cur.month, cur.day)
            end = datetime.datetime(nextday.year, nextday.month, nextday.day)
            num = ecgs_collection.count_documents({'timestamp': {'$gte': start, '$lt': end}})
            labels.append(cur.strftime('%d/%m/%Y'))
            values.append(num)
            cur = nextday
    else:
        if period == '7':
            for i in range(6, -1, -1):
                day = today - datetime.timedelta(days=i)
                start = datetime.datetime(day.year, day.month, day.day)
                end = start + datetime.timedelta(days=1)
                num = ecgs_collection.count_documents({'timestamp': {'$gte': start, '$lt': end}})
                labels.append(day.strftime('%d/%m'))
                values.append(num)
        elif period == '30':
            for i in range(29, -1, -1):
                day = today - datetime.timedelta(days=i)
                start = datetime.datetime(day.year, day.month, day.day)
                end = start + datetime.timedelta(days=1)
                num = ecgs_collection.count_documents({'timestamp': {'$gte': start, '$lt': end}})
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
                num = ecgs_collection.count_documents({'timestamp': {'$gte': start, '$lt': end}})
                labels.append(mdate.strftime('%m/%Y'))
                values.append(num)
        else:
            for i in range(6, -1, -1):
                day = today - datetime.timedelta(days=i)
                start = datetime.datetime(day.year, day.month, day.day)
                end = start + datetime.timedelta(days=1)
                num = ecgs_collection.count_documents({'timestamp': {'$gte': start, '$lt': end}})
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
    # suggest matching names first
    try:
        names = ecgs_collection.distinct('name', {'name': {'$regex': q, '$options': 'i'}})
        for n in names:
            if n and n not in suggestions:
                suggestions.append(n)
    except Exception:
        pass
    # also include any ObjectId hex strings that contain the query (scan recent docs)
    try:
        if len(suggestions) < 20:
            for d in ecgs_collection.find({}, {'_id': 1}).sort('timestamp', -1).limit(200):
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
    return render_template('view.html', ecg=rec)


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
    return render_template('print_ecg.html', ecg=rec)

@app.route('/delete/<ecg_id>')
def delete_ecg(ecg_id):
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return 'ID không hợp lệ', 400
    ecgs_collection.delete_one({'_id': oid})
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
        query['name'] = {'$regex': q, '$options': 'i'}
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
            query['timestamp'] = {'$gte': start, '$lt': end}
    else:
        # apply same time bounds as history chart
        if period == '7':
            start = datetime.datetime(today.year, today.month, today.day) - datetime.timedelta(days=6)
            query['timestamp'] = {'$gte': start}
        elif period == '30':
            start = datetime.datetime(today.year, today.month, today.day) - datetime.timedelta(days=29)
            query['timestamp'] = {'$gte': start}
        elif period == '12m':
            year = today.year
            month = today.month
            month -= 11
            while month <= 0:
                month += 12
                year -= 1
            start = datetime.datetime(year, month, 1)
            query['timestamp'] = {'$gte': start}
    docs = list(ecgs_collection.find(query).sort('timestamp', -1))
    # generate csv
    import csv
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['_id','name','timestamp','heart_rate','prediction'])
    for d in docs:
        cw.writerow([str(d['_id']), d.get('name',''), d.get('timestamp',''), d.get('heart_rate',''), d.get('prediction','')])
    output = si.getvalue()
    return output, 200, {'Content-Type':'text/csv', 'Content-Disposition':'attachment; filename="history.csv"'}

# --- API -----------------------------------------------------------
@app.route('/api/ecg/<ecg_id>')
def api_get_ecg(ecg_id):
    try:
        oid = ObjectId(ecg_id)
    except Exception:
        return jsonify({'error': 'invalid id'}), 400
    rec = ecgs_collection.find_one({'_id': oid})
    if not rec:
        return jsonify({'error':'not found'}), 404
    return jsonify({'signal': rec['signal']})

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
    result = analyze_signal(rec['signal'])
    ecgs_collection.update_one({'_id': oid}, {'$set': result})
    # notify user
    from flask import flash
    flash('Đã phân tích lại ECG. Kết quả mới đã cập nhật.', 'success')
    return redirect(url_for('view_ecg', ecg_id=ecg_id))

# legacy prediction (for frontend upload)
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    signal = data.get('signal', [])
    out = analyze_signal(signal)
    return jsonify(out)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
