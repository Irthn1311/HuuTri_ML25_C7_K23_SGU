from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None

    @abstractmethod
    def clean_data(self, df):
        """Sau khi load data, hàm clean_data này bao gồm các bước như:
            1. Xử lý missing values
            2. 
        """
        pass

    @abstractmethod
    def feature_engineering(self, df):
        pass

    @abstractmethod
    def train_model(self, X, y):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def run(self, df_train, df_test):
        print("🧹 Cleaning data...")
        df_train = self.clean_data(df_train)
        df_test = self.clean_data(df_test)

        print("🧩 Feature engineering...")
        df_train = self.feature_engineering(df_train)
        df_test = self.feature_engineering(df_test)

        X_train = df_train.drop('SalePrice', axis=1)
        y_train = df_train['SalePrice']

        print("🚀 Training model...")
        self.model = self.train_model(X_train, y_train)

        print("📊 Predicting on test set...")
        preds = self.predict(df_test)
        return preds
