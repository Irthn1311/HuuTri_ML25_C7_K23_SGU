import pandas as pd
import os

def load_data(data_cfg):
    """
    data_cfg: dictionary trong YAML chứa đường dẫn dữ liệu
    ví dụ:
      data:
        train: "data/raw/train.csv"
        test: "data/raw/test.csv"
        submission: "reports/experiments/submission.csv"
    """
    train_path = data_cfg.get("train", "data/raw/train.csv")
    test_path = data_cfg.get("test", "data/raw/test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu tại {train_path} hoặc {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test
