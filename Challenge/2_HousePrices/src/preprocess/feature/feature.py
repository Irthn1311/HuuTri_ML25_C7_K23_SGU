import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

class Feature:
    def __init__(self, df, target_col=None, task="auto"):
        """
        df: DataFrame
        target_col: tên cột target
        task: 'classification' hoặc 'regression' hoặc 'auto'
        """
        self.df = df.copy()
        self.target_col = target_col
        self.task = task or "auto"

    # --- 1. Univariate Selection (chi2) ---
    def univariate_selection(self, k=10):
        if self.target_col is None:
            raise ValueError("Cần target_col để chọn đặc trưng.")
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_encoded = pd.get_dummies(X, drop_first=True)
        selector = SelectKBest(score_func=chi2, k=min(k, X_encoded.shape[1]))
        X_new = selector.fit_transform(abs(X_encoded), y)

        selected_cols = X_encoded.columns[selector.get_support()]
        print(f"✅ Chọn {len(selected_cols)} đặc trưng theo chi2: {list(selected_cols)}")
        return pd.concat([pd.DataFrame(X_new, columns=selected_cols), y.reset_index(drop=True)], axis=1)

    # --- 2. Recursive Feature Elimination (RFE) ---
    def rfe_selection(self, k=10):
        if self.target_col is None:
            raise ValueError("Cần target_col để chọn đặc trưng.")
        X = pd.get_dummies(self.df.drop(columns=[self.target_col]), drop_first=True)
        y = self.df[self.target_col]

        if self.task == "classification":
            model = LogisticRegression(max_iter=1000)
        else:
            model = LinearRegression()

        rfe = RFE(model, n_features_to_select=min(k, X.shape[1]))
        X_rfe = rfe.fit_transform(X, y)
        selected_cols = X.columns[rfe.get_support()]
        print(f"🧩 RFE chọn {len(selected_cols)} đặc trưng: {list(selected_cols)}")
        return pd.concat([pd.DataFrame(X_rfe, columns=selected_cols), y.reset_index(drop=True)], axis=1)

    # --- 3. PCA (Feature Extraction) ---
    def pca_extraction(self, n_components=5):
        X = self.df.drop(columns=[self.target_col]) if self.target_col else self.df
        X_num = X.select_dtypes(include=np.number)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_num)
        cols = [f"PCA_{i+1}" for i in range(n_components)]
        print(f"🔹 PCA hoàn tất ({n_components} components, {np.sum(pca.explained_variance_ratio_):.2%} variance explained).")
        X_pca_df = pd.DataFrame(X_pca, columns=cols)
        if self.target_col:
            return pd.concat([X_pca_df, self.df[self.target_col].reset_index(drop=True)], axis=1)
        return X_pca_df

    # --- 4. Feature Importance (Random Forest) ---
    def feature_importance(self, top_n=10):
        if self.target_col is None:
            raise ValueError("Cần target_col để tính importance.")
        X = pd.get_dummies(self.df.drop(columns=[self.target_col]), drop_first=True)
        y = self.df[self.target_col]

        if self.task == "classification":
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = importances.sort_values(ascending=False).head(top_n)
        print(f"🔥 Top {top_n} feature quan trọng:\n{top_features}")
        return top_features

    # --- 5. Feature Engineering ---
    def add_feature(self, func, new_name=None):
        """
        func: có thể là hàm Python (callable) hoặc biểu thức dạng chuỗi (vd: "GrLivArea + GarageArea")
        new_name: tên cột mới
        """
        if isinstance(func, str):
            # Nếu người dùng truyền biểu thức, ta dùng pandas.eval
            try:
                self.df[new_name] = pd.eval(func, engine='python', local_dict=self.df)
            except Exception as e:
                raise ValueError(f"❌ Lỗi khi tính toán custom feature '{new_name}' với biểu thức: {func}\n{e}")
        elif callable(func):
            # Nếu người dùng truyền hàm, gọi trực tiếp
            self.df[new_name] = func(self.df)
        else:
            raise TypeError(f"❌ 'func' phải là callable hoặc string expression, nhưng nhận {type(func)}")

        print(f"✅ Đã thêm đặc trưng mới: {new_name}")
        return self.df

    # --- 6. Tổng hợp run ---
    def run(self, method="chi2", **kwargs):
        if method == "chi2":
            return self.univariate_selection(**kwargs)
        elif method == "rfe":
            return self.rfe_selection(**kwargs)
        elif method == "pca":
            return self.pca_extraction(**kwargs)
        elif method == "importance":
            return self.feature_importance(**kwargs)
        else:
            raise ValueError("Phương pháp feature không hợp lệ.")
