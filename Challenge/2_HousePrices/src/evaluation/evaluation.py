import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd

class Evaluation:
    def __init__(self, model, X_test, y_test, model_name="Model", task="auto"):
        """
        model: mô hình đã huấn luyện
        X_test, y_test: tập kiểm thử
        model_name: tên hiển thị
        task: "auto", "regression", hoặc "classification"
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        # Dự đoán
        self.y_pred = self.model.predict(X_test)
        self.y_prob = None
        if hasattr(model, "predict_proba"):
            self.y_prob = model.predict_proba(X_test)[:, 1]

        # Xác định loại bài toán
        if task == "auto":
            self.task = "classification" if len(np.unique(y_test)) <= 10 else "regression"
        else:
            self.task = task

    # === 1. Báo cáo đánh giá ===
    def report(self):
        print(f"\n🔹 Evaluation Report for {self.model_name} ({self.task})")

        if self.task == "classification":
            print(classification_report(self.y_test, self.y_pred))
            metrics = {
                "Accuracy": accuracy_score(self.y_test, self.y_pred),
                "Precision": precision_score(self.y_test, self.y_pred, zero_division=0),
                "Recall": recall_score(self.y_test, self.y_pred, zero_division=0),
                "F1-score": f1_score(self.y_test, self.y_pred, zero_division=0)
            }
            if self.y_prob is not None:
                metrics["ROC-AUC"] = roc_auc_score(self.y_test, self.y_prob)

        else:  # Regression
            y_true = np.maximum(self.y_test, 0)
            y_pred = np.maximum(self.y_pred, 0)
            try:
                metrics = {
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "RMSLE": np.sqrt(mean_squared_log_error(y_true, y_pred)),
                    "R²": r2_score(y_true, y_pred)
                }
            except ValueError:
                metrics = {
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "RMSLE": np.nan,
                    "R²": r2_score(y_true, y_pred)
                }
                print("⚠️ RMSLE bị bỏ qua vì dữ liệu có giá trị <= -1")

        print("\n Tổng hợp metrics:")
        print(pd.DataFrame(metrics, index=[self.model_name]).T)
        return metrics

    # === 2. Biểu đồ hồi quy ===
    def plot_regression_fit(self):
        if self.task != "regression":
            print(" Không áp dụng cho classification.")
            return
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], "r--")
        plt.title(f"Predicted vs Actual - {self.model_name}")
        plt.xlabel("Giá trị thực")
        plt.ylabel("Giá trị dự đoán")
        plt.show()

    # === 3. Confusion matrix (nếu là classification) ===
    def plot_confusion(self):
        if self.task != "classification":
            print(" Không áp dụng cho regression.")
            return
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # === 4. Feature importance (nếu có) ===
    def feature_importance(self, feature_names=None, top_n=15):
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importance))]
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False).head(top_n)

            plt.figure(figsize=(8, 5))
            sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
            plt.title(f"Top {top_n} Feature Importances - {self.model_name}")
            plt.show()
            return imp_df
        elif hasattr(self.model, "coef_"):
            coef = np.abs(self.model.coef_.flatten()) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": coef
            }).sort_values(by="Importance", ascending=False).head(top_n)
            plt.figure(figsize=(8, 5))
            sns.barplot(x="Importance", y="Feature", data=imp_df, palette="crest")
            plt.title(f"Top {top_n} Coefficients - {self.model_name}")
            plt.show()
            return imp_df
        else:
            print("⚠️ Mô hình không có thông tin feature importance.")
            return None

    # === 5. Tổng hợp toàn bộ ===
    def full_evaluation(self, feature_names=None):
        metrics = self.report()
        if self.task == "regression":
            self.plot_regression_fit()
        else:
            self.plot_confusion()
        self.feature_importance(feature_names)
        return metrics
