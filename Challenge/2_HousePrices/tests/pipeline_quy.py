from .base_pipeline import BasePipeline
from ..models.model_lightgbm import build_lightgbm
from ..features.feature_encoding import encode_categorical
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

class Pipeline_Quy(BasePipeline):
    def clean_data(self, df):
        """Làm sạch dữ liệu: loại bỏ hoặc điền missing values"""
        df = df.drop(columns=['Alley', 'PoolQC', 'Fence'], errors='ignore')
        df = df.fillna(df.median(numeric_only=True))
        return df

    def feature_engineering(self, df):
        """Tạo đặc trưng mới và encode các cột dạng categorical"""
        df = encode_categorical(df)
        df['TotalArea'] = df['GrLivArea'] + df.get('TotalBsmtSF', 0)
        df['OverallQuality'] = df['OverallQual'] * df['OverallCond']
        return df

    def train_model(self, X, y):
        """Chạy GridSearchCV với LightGBM (boosting)"""
        print("🔍 Running 10-Fold GridSearchCV...")

        model = build_lightgbm(self.config['model']['lightgbm'])

        # Grid search space
        param_grid = {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 300, 500],
        }

        # 10-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        grid = GridSearchCV(model, param_grid, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
        grid.fit(X, y)

        print(f"✅ Best params: {grid.best_params_}")
        print(f"✅ Best RMSE (CV): {-grid.best_score_:.4f}")

        # Train model cuối cùng với toàn bộ data, dùng best params
        best_model = build_lightgbm({**self.config['model']['lightgbm'], **grid.best_params_})
        best_model.fit(X, y)

        return best_model

    def predict(self, X_test):
        return self.model.predict(X_test)
