# 🏠 House Prices Prediction Project

## [Structure](https://github.com/drivendataorg/cookiecutter-data-science) (Project - Challenge)

## 📌 Mục tiêu

Xây dựng hệ thống pipeline tự động gồm các bước:
1. **EDA** — Khám phá và phân tích dữ liệu.
2. **Preprocess** — Làm sạch, chuẩn hóa, và chọn đặc trưng.
3. **Modeling** — Huấn luyện mô hình (LightGBM, Lasso, v.v.)
4. **Evaluation** — Đánh giá hiệu suất mô hình.
5. **Tracking** — Theo dõi và log kết quả thực nghiệm.
6. **Ensemble** — Kết hợp nhiều mô hình để cải thiện độ chính xác.

---

## 📂 Cấu trúc thư mục

```

2_HousePrices/
│
├── config/
│ └── global_config.yaml # Cấu hình toàn hệ thống (YAML)
│
├── data/
│ ├── raw/ # Dữ liệu gốc tải từ Kaggle
│ ├── interim/ # Dữ liệu sau bước xử lý tạm
│ └── processed/ # Dữ liệu sau khi tiền xử lý hoàn chỉnh
│
├── docs/ # Tài liệu mô tả, biểu đồ, note kỹ thuật
│
├── models/ # Nơi lưu model đã train (.pkl, .joblib)
│
├── notebooks/
│ ├── global_eda.ipynb # Notebook EDA tổng quan
│ └── experiments/ # Mỗi thành viên có notebook riêng (phát triển thử nghiệm)
│
├── reports/
│ ├── experiments/ # Kết quả thử nghiệm (metrics, logs)
│ ├── figures/ # Hình ảnh biểu đồ kết quả
│ └── results_summary.md # Tổng hợp kết quả model
│
├── src/
│ ├── eda/
│ │ └── eda.py # Class EDA (biểu đồ, phân tích thống kê)
│ │
│ ├── preprocess/
│ │ ├── Clean.py # Xử lý missing, duplicate, imbalance
│ │ ├── Preprocess.py # Chuẩn hóa, scale, normalize, binarize
│ │ └── Feature.py # Chọn và tạo đặc trưng
│ │
│ ├── model/
│ │ ├── LightGBM.py # Model LightGBM
│ │ └── Lasso.py # Model Lasso Regression
│ │
│ ├── evaluation/
│ │ └── Evaluation.py # Đánh giá model (MAE, RMSE, R², Confusion matrix, v.v.)
│ │
│ ├── ensemble/
│ │ └── Ensemble.py # Voting, Stacking, Bagging, Blending
│ │
│ ├── pipeline/
│ │ └── Pipeline.py # Luồng Clean → Preprocess → Feature → Model → Evaluate
│ │
│ ├── tracking/
│ │ └── tracking.py # Ghi log, metrics, lưu file kết quả
│ │
│ ├── utils/
│ │ └── data_loader.py # Load dữ liệu, tiện ích xử lý file
│ │
│ └── init.py # Đánh dấu package Python
│
├── tests/ # Unit tests cho module
│
├── SUBMIT-summarize_report/ # Tổng hợp kết quả cuối cùng để nộp Kaggle
│
├── main.py # Script chạy chính toàn pipeline
│
└── README.md # File hướng dẫn (bạn đang đọc)

```

## Kết quả sẽ được lưu:

- Model: models/<model_name>.pkl
- Báo cáo: reports/experiments/<experiment_name>/metrics.json
- Hình ảnh: reports/figures/

## ⚙️ Run: (console)

- Example: 
`python main.py --config config/config_basic-test_quy.yaml`


## 🔗 Code Flow

```text
                        ┌──────────────────────────────┐
                        │  main.py / train.py          │
                        │  (entrypoint - CLI call)     │
                        └────────────┬─────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │  Đọc file config.yaml        │
                        │  (setup toàn bộ pipeline)    │
                        └────────────┬─────────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────────┐
                        │  src/pipeline/               │
                        │  (chạy end-to-end pipeline)  │
                        └────────────┬─────────────────┘
                                     │
       ┌─────────────────────────────┼──────────────────────────────┐
       ▼                             ▼                              ▼
┌───────────────┐          ┌──────────────────────┐        ┌────────────────────────┐
│ load_data()   │          │ preprocess/cleaning │        │ build_features()        │
│ (raw → df)    │          │ - xử lý missing     │        │ - tạo đặc trưng mới     │
│ - đọc train   │          │ - remove duplicate  │        │ - encode categorical     │
│ - đọc test    │          │ - scale/normalize   │        │ - feature selection (RFE)│
└───────────────┘          └──────────────────────┘        └────────────────────────┘
       │                             │                              │
       └─────────────────────────────┼──────────────────────────────┘
                                     ▼
                         ┌────────────────────────────┐
                         │   model training / tuning  │
                         │   (LightGBM, Lasso, ...)   │
                         └────────────┬───────────────┘
                                      │
                                      ▼
                         ┌────────────────────────────┐
                         │     evaluation & report    │
                         │  - metrics (MAE, RMSE, R²) │
                         │  - feature importance      │
                         └────────────┬───────────────┘
                                      │
                                      ▼
                         ┌────────────────────────────┐
                         │ tracking & submission      │
                         │ - log kết quả              │
                         │ - lưu submission.csv       │
                         └────────────────────────────┘


