from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    raw_data_path: str
    processed_data_path: str
    features: str
    features_path: str
    experiments_path: str
    random_seed: int

    def __init__(self):
        self.raw_data_path = Path("data/raw/LibriTTS")
        self.processed_data_path = Path("data/processed/audio")
        self.features = "features.csv"
        self.features_path = self.processed_data_path / ".." / self.features
        self.experiments_path = Path("experiments/")
        self.random_seed = 1312
