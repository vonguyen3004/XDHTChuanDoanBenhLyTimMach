import pandas as pd

path = r"e:\XDHTChuanDoanBenhLyTimMach\ml\ECG_Diagnosis_System\mitbih_train.csv"
df = pd.read_csv(path, header=None)
# extract the first row as a 1×N DataFrame so to_csv writes one line
rowdf = df.iloc[[0]]
rowdf.to_csv('upload_sample.csv', index=False, header=False)
print('rewrote upload_sample.csv with', rowdf.shape[1], 'values')
