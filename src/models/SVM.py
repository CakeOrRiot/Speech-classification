from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVM:
    def __init__(self, svm_params=None, scaler_params=None):
        if svm_params is None:
            svm_params = {}
        if scaler_params is None:
            scaler_params = {}
        self.model = SVC(**svm_params)
        self.scaler = StandardScaler(**scaler_params)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def get_params(self, deep=False):
        return {
            "SVM": self.model.get_params(),
            "StandartScaler": self.scaler.get_params(),
        }
