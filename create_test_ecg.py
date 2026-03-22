from pymongo import MongoClient
from datetime import datetime

client = MongoClient('mongodb://localhost:27017/')
db = client['ecg_diagnosis_db']
ecg_coll = db['ecg_records']

test_rec = {
    'ecg_file': 'test_sample.csv',
    'name': 'Test ECG Sample',
    'timestamp': datetime.now(),
    'bpm': 75,
    'heart_rate': 75,
    'prediction': 0,
    'prediction_label': 'IMI',
    'prediction_confidence': 0.95,
    'prediction_desc': 'Nhồi máu cơ tim thành dưới',
    'patient': {
        'patient_id': 'P001',
        'name': 'Nguyễn Văn A',
        'age': 45,
        'gender': 'Nam',
        'phone': '0901234567',
        'address': 'Hà Nội',
        'notes': 'Test patient'
    },
    'multi_label_predictions': [
        {'disease': 'IMI', 'confidence': 0.95},
        {'disease': 'LBBB', 'confidence': 0.45}
    ]
}

result = ecg_coll.insert_one(test_rec)
print(f'Inserted: {result.inserted_id}')
