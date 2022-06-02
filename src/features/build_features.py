import os
import pandas as pd
import numpy as np
import librosa
from src.features.mean import mean
from src.features.mfcc import mfcc


def build_features_for_sample(file_path):
    features = []

    signal, sr = librosa.load(file_path)
    features.extend(mean(signal))
    features.extend(mfcc(signal, sr))
    return features


def build_features(data_path, labels):
    X = []
    y = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            reader_id = int(file.split("_")[0])
            label = labels[reader_id]
            y.append(label)
            X.append(build_features_for_sample(os.path.join(root, file)))
    return np.array(X), np.array(y)


def create_df_features(data_path, labels, output="features.csv"):
    X, y = build_features(data_path, labels)
    columns = ["mean"] + [f"mfcc_{i}" for i in range(X.shape[1] - 1)]
    df = pd.DataFrame(X, columns=columns)
    df["gender"] = y
    df["gender"] = df["gender"].map({"F": 0, "M": 1})
    df.to_csv(data_path / ".." / output)
    return df
