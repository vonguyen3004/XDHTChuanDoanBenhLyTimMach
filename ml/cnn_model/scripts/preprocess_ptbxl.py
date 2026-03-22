import os
import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_ROOT = os.path.join(BASE_DIR, 'data')
ptbxl_path = os.path.join(DATA_ROOT, 'ptbxl')
# If you have moved dataset to data/ECG_Diagnosis_System (old mitbih) then this is separate.

sampling_rate = 500

# Load metadata
df = pd.read_csv(os.path.join(ptbxl_path, 'ptbxl_database.csv'))
df_ptbxl_statement = pd.read_csv(os.path.join(ptbxl_path, 'scp_statements.csv'), index_col=0)

# Map to diagnostic subclass
df_ptbxl_statement = df_ptbxl_statement[df_ptbxl_statement.diagnostic == 1]
agg_df = df_ptbxl_statement.groupby('diagnostic_class').agg({'diagnostic_subclass': lambda x: list(set(x))})

# Load and convert annotation data
Y = pd.DataFrame(0, index=df.index, columns=agg_df.index)
for i in range(len(df)):
    if pd.isna(df.scp_codes.iloc[i]):
        continue
    codes = ast.literal_eval(df.scp_codes.iloc[i])
    for code in codes.keys():
        if code in agg_df.index:
            Y.loc[i, code] = codes[code]

# Encode labels (multi-label to binary)
label_encoder = LabelEncoder()
Y_encoded = Y.apply(lambda x: label_encoder.fit_transform(x), axis=0)  # Adjust for multi-label

# Load signals
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 500:
        for record in df.filename_hr:
            print("Using ECG files:", record)
            print("Sampling rate:", 500)
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    data = np.array([signal for signal, meta in data])
    return data

X = load_raw_data(df, sampling_rate, ptbxl_path)

# Normalize signals (zero-mean, unit variance per lead)
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

# Split by folds (1-8 train, 9 val, 10 test)
train_folds = list(range(1, 9))
val_fold = 9
test_fold = 10

X_train = X[df.strat_fold.isin(train_folds)]
y_train = Y_encoded[df.strat_fold.isin(train_folds)]

X_val = X[df.strat_fold == val_fold]
y_val = Y_encoded[df.strat_fold == val_fold]

X_test = X[df.strat_fold == test_fold]
y_test = Y_encoded[df.strat_fold == test_fold]

# Save processed data
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("Preprocessing complete. Shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")