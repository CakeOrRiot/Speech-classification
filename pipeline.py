import os
from src.setup import setup
from src.processing.process import process
from src.config import Config
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from src.models.DecisionTree import DecisionTree
from src.models.SVM import SVM
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import datetime

config = Config()


def pipeline(
    *, need_setup=True, need_process=True, features=None, model=None, model_name=None
):
    if need_setup:
        print("Downloading data")
        setup()
    if need_process:
        print("Processing data")
        process(features=features)
    if model is not None:
        if model_name is None:
            model_name = type(model).__name__
        print(f"Running experiment with {model_name}")

        score = experiment(model, model_name)


def log_experiment(model, train_score, test_score, model_name=""):
    file_name = (
        f"{datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}_{model_name}.txt"
    )
    with open(config.experiments_path / file_name, "wt", encoding="utf-8") as f:
        print(
            model_name,
            f"Params: {model.get_params()}",
            f"Train: {train_score}",
            f"Test: {test_score}",
            str(config),
            sep="\n",
            file=f,
        )


def evaluate(model, data, true_labels):
    pred_labels = model.predict(data)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def experiment(model, model_name=""):
    df = pd.read_csv(config.features_path)
    readers = pd.unique(df["reader"])
    readers_train, readers_test = train_test_split(
        readers, test_size=0.4, random_state=config.random_seed
    )

    df_train = df[df["reader"].isin(readers_train)]
    df_test = df[df["reader"].isin(readers_test)]
    X_train = np.array(df_train.drop(columns=["gender", "reader"]))
    X_test = np.array(df_test.drop(columns=["gender", "reader"]))
    y_train = np.array(df_train["gender"])
    y_test = np.array(df_test["gender"])

    model.fit(X_train, y_train)
    train_score = evaluate(model, X_train, y_train)
    test_score = evaluate(model, X_test, y_test)
    log_experiment(model, train_score, test_score, model_name=model_name)
    return train_score, test_score


if __name__ == "__main__":
    pipeline(
        # features="features.csv",
    )
    pipeline(
        need_setup=False,
        need_process=False,
        features="features.csv",
        model=DecisionTree(),
    )
    pipeline(
        need_setup=False,
        need_process=False,
        features="features.csv",
        model=GridSearchCV(
            DecisionTree(),
            param_grid={"max_depth": range(3, 10), "min_samples_leaf": range(1, 30)},
        ),
        model_name="DecisionTreeWithRegularization",
    )
    pipeline(need_setup=False, need_process=False, features="features.csv", model=SVM())
