import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from src.evaluation import evaluation

class ModelLightGBM:
    def __init__(self, params=None, param_grid=None, random_state=42, test_size=0.2, model_name="lightgbm"):
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.model = None
        self.best_params = None
        self.metrics = {}

        self.params = params or {"n_estimators": 1000, "learning_rate": 0.05, "random_state": self.random_state}
        self.param_grid = param_grid or {
            "num_leaves": [31, 50, 70],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 500, 1000]
        }
        os.makedirs("experiments/models", exist_ok=True)

    def prepare_data(self, df, target_col):
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def tune_params(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=lgb.LGBMRegressor(**self.params),
            param_grid=self.param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            # Using all CPU 
            n_jobs=-1, # number of threads run in parallel 
            
            # verbose: 0-quite, 1-short, 2-detail
            verbose=1, 
        )
        grid.fit(X_train, y_train)
        self.best_params = grid.best_params_
        return self.best_params

    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        best_params = self.tune_params(X_train, y_train)
        self.model = lgb.LGBMRegressor(**best_params, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        self.evaluate(X_test, y_test, feature_names=X_train.columns)
        self.save_model()
        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = evaluation.Evaluation(self.model, X_test, y_test, model_name=self.model_name, task="regression")
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def predict(self, X_new):
        X_new = pd.get_dummies(X_new, drop_first=True)
        return self.model.predict(X_new)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        return path
