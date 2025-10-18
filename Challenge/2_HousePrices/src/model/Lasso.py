import os
import joblib
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from Evaluation.Evaluation import Evaluation

class ModelLasso:
    def __init__(self, alphas=None, random_state=42, test_size=0.2, model_name="lasso"):
        self.alphas = alphas or [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.model = None
        self.best_alpha = None
        self.metrics = {}
        os.makedirs("experiments/models", exist_ok=True)

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def tune_alpha(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=Lasso(random_state=self.random_state),
            param_grid={"alpha": self.alphas},
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        self.best_alpha = grid.best_params_["alpha"]
        return self.best_alpha

    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        best_alpha = self.tune_alpha(X_train, y_train)
        self.model = Lasso(alpha=best_alpha, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        self.evaluate(X_test, y_test, feature_names=X_train.columns)
        self.save_model()
        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task="regression")
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        return self.model.predict(X_new)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        return path
