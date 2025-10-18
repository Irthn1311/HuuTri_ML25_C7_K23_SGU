import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, Normalizer, Binarizer
)

class Preprocess:
    def __init__(self, df):
        """
        df: DataFrame cần xử lý
        """
        self.df = df.copy()

    # --- 1. Rescale (MinMaxScaler) ---
    def rescale(self, feature_range=(0, 1)):
        scaler = MinMaxScaler(feature_range=feature_range)
        num_cols = self.df.select_dtypes(include="number").columns
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        print(f"📏 Rescale hoàn tất (range={feature_range})")
        return self.df

    # --- 2. Standardize (Z-score) ---
    def standardize(self):
        scaler = StandardScaler()
        num_cols = self.df.select_dtypes(include="number").columns
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        print("✅ Standardize hoàn tất (Z-score).")
        return self.df

    # --- 3. Normalize (vector norm) ---
    def normalize(self, norm="l2"):
        normalizer = Normalizer(norm=norm)
        num_cols = self.df.select_dtypes(include="number").columns
        self.df[num_cols] = normalizer.fit_transform(self.df[num_cols])
        print(f"🧭 Normalize hoàn tất (norm={norm}).")
        return self.df

    # --- 4. Binarize ---
    def binarize(self, threshold=0.0):
        binarizer = Binarizer(threshold=threshold)
        num_cols = self.df.select_dtypes(include="number").columns
        self.df[num_cols] = binarizer.fit_transform(self.df[num_cols])
        print(f"🔹 Binarize hoàn tất (threshold={threshold}).")
        return self.df

    # --- 5. Tổng hợp run ---
    def run(self, method="standardize", **kwargs):
        """
        method: rescale | standardize | normalize | binarize
        """
        if method == "rescale":
            return self.rescale(**kwargs)
        elif method == "standardize":
            return self.standardize()
        elif method == "normalize":
            return self.normalize(**kwargs)
        elif method == "binarize":
            return self.binarize(**kwargs)
        else:
            raise ValueError("Phương pháp không hợp lệ.")
