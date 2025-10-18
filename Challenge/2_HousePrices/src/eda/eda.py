import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, df):
        """Khởi tạo DataFrame"""
        self.df = df
        sns.set(style="whitegrid", palette="pastel")

    # Tổng quan dữ liệu
    def overview(self):
        print(" Kích thước dữ liệu:", self.df.shape)
        print("\n Kiểu dữ liệu:")
        print(self.df.dtypes)
        print("\n Thống kê mô tả:")
        display(self.df.describe(include='all').T)

    # Kiểm tra giá trị thiếu
    def missing_values(self):
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            print("✅ Không có giá trị thiếu.")
            return
        print("📉 Các cột có giá trị thiếu:\n", missing)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing.index, y=missing.values, color='salmon')
        plt.title("Số lượng giá trị thiếu theo cột")
        plt.xticks(rotation=45)
        plt.show()

    # Phân bố từng biến (Histogram + KDE)
    def distribution(self, column):
        plt.figure(figsize=(6, 4))
        sns.histplot(self.df[column], kde=True, bins=30, color='skyblue')
        plt.title(f"Phân bố của {column}")
        plt.show()

    def box_violin(self, column, kind="box", by=None):
        """
        kind: 'box' hoặc 'violin'
        """
        plt.figure(figsize=(6, 4))
        if kind == "box":
            if by:
                sns.boxplot(x=self.df[by], y=self.df[column])
                plt.title(f"Boxplot {column} theo {by}")
            else:
                sns.boxplot(y=self.df[column])
                plt.title(f"Boxplot của {column}")
        elif kind == "violin":
            if by:
                sns.violinplot(x=self.df[by], y=self.df[column], inner="quartile")
                plt.title(f"Violin plot {column} theo {by}")
            else:
                sns.violinplot(y=self.df[column], inner="quartile")
                plt.title(f"Violin plot của {column}")
        plt.show()

    # Biểu đồ tương quan
    def correlation_matrix(self):
        corr = self.df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Ma trận tương quan (Correlation Matrix)")
        plt.show()

    # Biểu đồ cặp (Pairplot)
    def pairplot(self, columns=None, target=None):
        if target and columns:
            sns.pairplot(self.df[columns + [target]], hue=target)
        elif columns:
            sns.pairplot(self.df[columns])
        else:
            sns.pairplot(self.df)
        plt.suptitle("Biểu đồ tương quan cặp (Pairplot)", y=1.02)
        plt.show()

    # Biểu đồ phân loại (Countplot)
    def categorical_summary(self, column, target=None):
        plt.figure(figsize=(7, 4))
        sns.countplot(x=self.df[column], palette="pastel")
        plt.title(f"Tần suất của {column}")
        plt.xticks(rotation=45)
        plt.show()

    # Scatterplot giữa 2 biến
    def scatterplot(self, x, y, hue=None):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=self.df[x], y=self.df[y], hue=self.df[hue] if hue else None, palette="coolwarm")
        plt.title(f"Biểu đồ phân tán {x} vs {y}")
        plt.show()

    # Kiểm tra ngoại lai (IQR)
    def outlier_detection(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[column] < lower) | (self.df[column] > upper)]
        print(f" Phát hiện {len(outliers)} ngoại lai trong {column}")
        self.box_violin(column)
        return outliers

    # Kiểm tra skewness
    def check_skewness(self):
        skewed = self.df.skew(numeric_only=True).sort_values(ascending=False)
        print(" Mức độ lệch (Skewness):")
        print(skewed)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=skewed.index, y=skewed.values, palette="coolwarm")
        plt.title("Độ lệch của các biến số")
        plt.xticks(rotation=45)
        plt.show()

    # Dashboard cho biến số
    def numeric_dashboard(self):
        num_cols = self.df.select_dtypes(include=np.number).columns
        for col in num_cols:
            print(f"\n --- {col} ---")
            self.distribution(col)
            self.box_violin(col)

    # Dashboard cho biến phân loại
    def categorical_dashboard(self):
        cat_cols = self.df.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            print(f"\n --- {col} ---")
            self.categorical_summary(col)
