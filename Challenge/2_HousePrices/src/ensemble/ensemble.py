import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor, StackingClassifier, StackingRegressor
from sklearn.model_selection import train_test_split
from evaluation.Evaluation import Evaluation

class Ensemble:
    def __init__(self, models, task="auto", method="voting", meta_model=None, test_size=0.2, random_state=42):
        """
        models: list of (name, model) tuples
        task: classification | regression | auto
        method: voting | stacking | bagging | blending
        meta_model: model meta (nếu stacking/blending)
        """
        self.models = models
        self.task = task
        self.method = method
        self.meta_model = meta_model
        self.test_size = test_size
        self.random_state = random_state
        self.ensemble_model = None
        self.metrics = {}

    # === 1. Build ensemble ===
    def build(self):
        if self.method == "voting":
            if self.task == "classification":
                self.ensemble_model = VotingClassifier(estimators=self.models, voting="soft")
            else:
                self.ensemble_model = VotingRegressor(estimators=self.models)

        elif self.method == "stacking":
            if self.task == "classification":
                self.ensemble_model = StackingClassifier(estimators=self.models, final_estimator=self.meta_model)
            else:
                self.ensemble_model = StackingRegressor(estimators=self.models, final_estimator=self.meta_model)

        elif self.method == "bagging":
            base_model = self.models[0][1]
            if self.task == "classification":
                self.ensemble_model = BaggingClassifier(base_model, n_estimators=10, random_state=self.random_state)
            else:
                self.ensemble_model = BaggingRegressor(base_model, n_estimators=10, random_state=self.random_state)
        else:
            raise ValueError("Phương pháp ensemble không hợp lệ.")

        print(f"✅ Đã tạo ensemble theo kiểu {self.method}.")
        return self.ensemble_model

    # === 2. Train + Evaluate ===
    def train_evaluate(self, df, target_col):
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        model = self.build()
        model.fit(X_train, y_train)

        evaluator = Evaluation(model, X_test, y_test, model_name="EnsembleModel", task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=X.columns)
        return model, self.metrics
