# ======================================================
# main.py  —  Entry point của dự án ML House Prices
# Run:
#   python main.py --config config/config.yaml
# Example:
#   python main.py --config config/config_basic-test_quy.yaml
# ======================================================

import argparse
import yaml
import os
import pandas as pd
from src.utils.data_loader import load_data
from src.pipeline.pipeline import Pipeline


def main():
    # -------------------------------
    # 1️⃣ Parse arguments từ CLI
    # -------------------------------
    parser = argparse.ArgumentParser(description="🏗️ Run ML Pipeline for House Prices Project")
    parser.add_argument(
        "--config",
        required=True,
        help="Đường dẫn đến file config YAML (vd: config/config_quy.yaml)"
    )
    args = parser.parse_args()

    # -------------------------------
    # 2️⃣ Đọc file cấu hình
    # -------------------------------
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"❌ Không tìm thấy file config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"\n📘 Loaded config from: {args.config}")

    # -------------------------------
    # 3️⃣ Kiểm tra cấu hình dữ liệu
    # -------------------------------
    data_cfg = config.get("data", None)
    if data_cfg is None:
        raise ValueError("❌ Thiếu phần 'data' trong config YAML.")

    train_path = data_cfg.get("train")
    test_path = data_cfg.get("test")
    sub_path = data_cfg.get("submission", "reports/submission.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {train_path} hoặc {test_path}")

    # -------------------------------
    # 4️⃣ Load dữ liệu
    # -------------------------------
    df_train, df_test = load_data(data_cfg)
    print(f"📊 Loaded data: train={df_train.shape}, test={df_test.shape}")

    # -------------------------------
    # 5️⃣ Khởi tạo Pipeline
    # -------------------------------
    target_col = config.get("target_col", None)
    if target_col is None:
        raise ValueError("⚠️ 'target_col' chưa được khai báo trong file config.")

    task = config.get("task", "auto")

    pipeline = Pipeline(
        df=df_train,
        target_col=target_col,
        task=task,
        config=config
    )

    # -------------------------------
    # 6️⃣ Chạy toàn bộ Pipeline
    # -------------------------------
    print("\n🚀 Bắt đầu chạy end-to-end pipeline...\n")
    model, metrics = pipeline.run()

    # -------------------------------
    # 7️⃣ Tạo Submission (nếu có model.predict)
    # -------------------------------
    if hasattr(model, "predict"):
        preds = model.predict(pd.get_dummies(df_test, drop_first=True))
        sub = pd.DataFrame({
            "Id": df_test["Id"],
            "SalePrice": preds
        })
        os.makedirs(os.path.dirname(sub_path), exist_ok=True)
        sub.to_csv(sub_path, index=False)
        print(f"\n✅ Done! Submission saved to: {sub_path}")
    else:
        print("\n⚠️ Model không có phương thức predict(), bỏ qua phần tạo submission.")

    # -------------------------------
    # 8️⃣ In kết quả đánh giá
    # -------------------------------
    print("\n📈 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")


if __name__ == "__main__":
    main()
