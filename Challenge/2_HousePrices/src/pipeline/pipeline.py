import pandas as pd
from src.preprocess.clean.clean import Clean
from src.preprocess.preprocess.preprocess import Preprocess
from src.preprocess.feature.feature import Feature
from src.model.LightGBM import ModelLightGBM
from src.evaluation.evaluation import Evaluation

class Pipeline:
    def __init__(self, df, target_col, task="auto", model_class=ModelLightGBM, config=None):
        """
        df: dữ liệu đầu vào
        target_col: cột target
        task: classification | regression | auto
        model_class: class model sẽ dùng (LightGBM, XGB, RF,...)
        config: dict chứa cấu hình cho toàn pipeline
        """
        self.df = df.copy()
        self.target_col = target_col
        self.task = task
        self.model_class = model_class
        self.config = config or {}

        self.model = None
        self.metrics = {}

    # === 1. Clean ===
    def step_clean(self):
        cfg = self.config.get("clean", {})
        cleaner = Clean(
            self.df,
            target_col=self.target_col,
            balance_method=cfg.get("balance_method"),
        )
        self.df = cleaner.run(
            remove_dup=cfg.get("remove_dup", True),
            handle_na=cfg.get("handle_na", True),
            balance=cfg.get("balance", False)
        )
        print("✅ Hoàn tất bước Clean.")
        return self.df

    # === 2. Preprocess ===
    def step_preprocess(self):
        cfg = self.config.get("preprocess", {})
        pre = Preprocess(self.df)
        self.df = pre.run(
            method=cfg.get("method", "standardize"),
            **cfg.get("params", {})
        )
        print("✅ Hoàn tất bước Preprocess.")
        return self.df

    # === 3. Feature ===
    def step_feature(self):
        cfg = self.config.get("feature", {})
        feat = Feature(self.df, target_col=self.target_col, task=self.task)

        # Feature selection/extraction
        if cfg.get("method"):
            self.df = feat.run(method=cfg["method"], **cfg.get("params", {}))

        # Feature engineering (optional custom func)
        if cfg.get("custom_func"):
            self.df = feat.add_feature(
                func=cfg["custom_func"],
                new_name=cfg.get("custom_name")
            )
        print("✅ Hoàn tất bước Feature.")
        return self.df

    # === 4. Train model ===
    def step_train_model(self):
        cfg = self.config.get("model", {})
        # Check for target column
        # if self.target_col not in self.df.columns:
            # raise ValueError(f"❌ Missing target column '{self.target_col}' before model training.")
        model = self.model_class(
            params=cfg.get("params"),
            param_grid=cfg.get("param_grid"),
            random_state=cfg.get("random_state", 42),
            test_size=cfg.get("test_size", 0.2),
            model_name=cfg.get("model_name", "LightGBM")
        )
        self.model = model.train(self.df, target_col=self.target_col)
        self.metrics = model.metrics
        print("✅ Train model hoàn tất.")
        return self.model, self.metrics

    # === 5. Evaluate ===
    def step_evaluate(self):
        X = pd.get_dummies(self.df.drop(columns=[self.target_col]), drop_first=True)
        y = self.df[self.target_col]
        evaluator = Evaluation(self.model, X, y, model_name="FinalModel", task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=X.columns)
        print("📊 Đánh giá hoàn tất.")
        return self.metrics

    # === 6. Run toàn bộ pipeline ===
    def run(self):
        print("\n🚀 Bắt đầu pipeline ML...\n")
        self.step_clean()
        self.step_preprocess()
        self.step_feature()
        self.step_train_model()
        print("\n🎯 Pipeline hoàn tất!\n")
        return self.model, self.metrics
