import pandas as pd, os
paths = [r"e:\XDHTChuanDoanBenhLyTimMach\ml\ECG_Diagnosis_System\mitbih_train.csv",
         r"e:\XDHTChuanDoanBenhLyTimMach\ml\ECG_Diagnosis_System\mitbih_test.csv"]
for path in paths:
    print(path, os.path.exists(path))
    df = pd.read_csv(path, header=None)
    print(df.shape)
    print('head', df.iloc[0,:5].tolist())
    print('tail', df.iloc[0,-5:].tolist())
