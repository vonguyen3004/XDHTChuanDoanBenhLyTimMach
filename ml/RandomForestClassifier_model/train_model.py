"""Simple training placeholder for ECG model."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


import pandas as pd

def load_data(csv_path=None):
    """Load training data.

    If ``csv_path`` is provided and exists, it should point to a file
    where each row corresponds to one ECG example.  The last column is
    treated as the label and the preceding columns are signal samples.
    This matches the MIT‑BIH format that has 187 signal values plus one
    class code (0–4).

    Otherwise fall back to a tiny synthetic dataset for quick sanity checks.
    """
    if csv_path and os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        # last column is label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].astype(int).values
        return X, y

    # synthetic data (default)
    X = np.random.randn(1000, 100)
    y = np.random.randint(0, 2, size=1000)
    return X, y


def train(csv_path=None):
    X, y = load_data(csv_path)
    if X.size == 0:
        raise ValueError("No data available for training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
    # save model into ml/randomForestClassifier folder; compute path relative to this script file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'randomForestClassifier'))
    os.makedirs(base_dir, exist_ok=True)
    output_path = os.path.join(base_dir, 'RandomForestClassifier_model.joblib')
    joblib.dump(clf, output_path)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train ECG classification model')
    parser.add_argument('csv', nargs='?',
                        default='data/ECG_Diagnosis_System/mitbih_train.csv',
                        help='path to CSV dataset (rows: samples, last column=label)')
    args = parser.parse_args()

    # If this script is run from outside repo root, resolve relative to repo root
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        cwd = os.path.abspath(os.getcwd())
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if not os.path.exists(csv_path):
            candidate = os.path.join(repo_root, csv_path)
            if os.path.exists(candidate):
                csv_path = candidate

    train(csv_path)
