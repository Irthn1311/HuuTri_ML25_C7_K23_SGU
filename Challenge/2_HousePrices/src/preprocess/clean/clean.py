import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class Clean:
    def __init__(self, df, target_col=None, balance_method=None, random_state=42):
        """
        df: DataFrame ban đầu
        target_col: cột nhãn (nếu cần cân bằng)
        balance_method: 'smote', 'undersample', hoặc None
        """
        self.df = df.copy()
        self.target_col = target_col
        self.balance_method = balance_method
        self.random_state = random_state

    # --- 1. Xử lý giá trị trùng lặp ---
    def remove_duplicates(self, keep='first'):
        before = len(self.df)
        self.df = self.df.drop_duplicates(keep=keep)
        print(f"🧩 Đã xóa {before - len(self.df)} dòng trùng lặp.")
        return self.df

    # --- 2. Xử lý giá trị thiếu ---
    def handle_missing(self, strategy="mean", fill_value=None):
        """
        strategy: mean | median | mode | fill
        """
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
            if strategy == "mean" and self.df[col].dtype != 'O':
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == "median" and self.df[col].dtype != 'O':
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == "mode":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif strategy == "fill":
                self.df[col].fillna(fill_value, inplace=True)
        print("✅ Đã xử lý giá trị thiếu theo chiến lược: ", strategy)
        return self.df

    # --- 3. Cân bằng dữ liệu ---
    def balance_data(self):
        if not self.target_col or not self.balance_method:
            print("⚠️ Không có target_col hoặc balance_method, bỏ qua cân bằng.")
            return self.df

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        if self.balance_method == "smote":
            sampler = SMOTE(random_state=self.random_state)
        elif self.balance_method == "undersample":
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            print("⚠️ balance_method không hợp lệ.")
            return self.df

        X_res, y_res = sampler.fit_resample(X, y)
        self.df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=self.target_col)], axis=1)
        print(f"✅ Đã cân bằng dữ liệu bằng {self.balance_method}.")
        return self.df

    # --- 4. Tổng hợp ---
    def run(self, remove_dup=True, handle_na=True, balance=True):
        if remove_dup:
            self.remove_duplicates()
        if handle_na:
            self.handle_missing()
        if balance:
            self.balance_data()
        return self.df
