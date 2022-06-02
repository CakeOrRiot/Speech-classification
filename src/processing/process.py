import pandas as pd
import os
import shutil
from src.config import Config
from src.features.build_features import create_df_features


def process(features=None, config=None):
    if config is None:
        config = Config()
    speakers = pd.read_csv(
        config.raw_data_path / "speakers.tsv", sep="\t", index_col=False
    )
    speakers.to_csv(config.processed_data_path / ".." / "speakers.csv", index=False)

    for subdir, dirs, files in os.walk(config.raw_data_path / "dev-clean"):
        for file in files:
            file_path = subdir + os.sep + file
            if file.endswith(".wav"):
                shutil.copy2(file_path, config.processed_data_path / file)
    print("Copy done")
    if features is None:
        print("Creating features")
        labels = dict(zip(speakers["READER"], speakers["GENDER"]))
        create_df_features(config.processed_data_path, labels)
    else:
        shutil.copy2(features, config.processed_data_path / ".." / "features.csv")
