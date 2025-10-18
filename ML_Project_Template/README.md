# [Structure](https://github.com/drivendataorg/cookiecutter-data-science) (Project - Challenge)

```
ML_Project_Template/
│
├── data/                    <- Chứa dữ liệu qua từng giai đoạn xử lí, thử nghiệm  (.csv, .parquet, .pkl)
│   ├── raw/                 <- Dữ liệu gốc (immutable)
│   ├── interim/             <- Dữ liệu trung gian (đã transform/clean một phần)
│   └── processed/           <- Dữ liệu sẵn sàng cho mô hình
│
├── notebooks/               <- nơi thực nghiệm features, model,... test, debugs
│   ├── 0.0-overview.ipynb
│   ├── 1.0-eda.ipynb
│   ├── 2.0-data_cleaning.ipynb
│   ├── 3.0-feature_engineering.ipynb
│   ├── 4.0-model_baseline_<name>.ipynb
│   └── 5.0-final_model.ipynb
│
├── reports/                 <- Chứa báo cáo, hình ảnh, submission, log kết quả (.md, .png, .csv, .pdf)
│   ├── figures/             <- Biểu đồ, hình ảnh
│   ├── experiments/         <- Báo cáo thực nghiệm, kết quả từng model
│   └── results_summary.md   <- Tổng kết kết quả so sánh các mô hình
│
├── models/                  <- Chứa mô hình đã huấn luyện, kết quả tuning, ensemble
│   ├── trained/             <- File model (.pkl, .joblib)
│   ├── logs/                <- Log huấn luyện, tensorboard logs
│   └── registry.json        <- Theo dõi model versions và metadata
│
├── src/                     <- nơi chạy end-to-end project (load data → preprocess → train → evaluate → save model)
│   ├── data/
│   │   ├── load_data.py
│   │   ├── cleaning.py
│   │   ├── preprocess.py
│   ├── features/
│   │   ├── feature_selection.py
│   │   ├── feature_extraction.py
│   │   └── feature_engineering.py
│   ├── modeling/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── tune.py
│   │   ├── ensemble.py
│   │   └── predict.py
│   ├── tracking/
│   │   ├── experiment_tracker.py  <- MLflow hoặc wandb integration
│   │   └── metrics_logger.py
│   │
│   └── pipeline.py
│
├── team/                 <- Dành cho làm việc nhóm
│   ├── Readme.md         <- Danh sách thành viên
│   └── progress_logs/    <- Nhật ký cá nhân, tiến độ từng thành viên
│
├── train.py                 <- file entrypoint chính (CLI)
│
├── config.yaml               # cấu hình pipeline (đường dẫn, tham số)
│
├── docs/                    <- Tài liệu chi tiết, hướng dẫn triển khai, API, pipeline, v.v.
│
├── references               <- Data dictionaries, manuals, and all other explanatory materials.
│
└── README.md                <- Giới thiệu dự án, mô tả mục tiêu, hướng dẫn cài đặt & chạy. 
```

## [Explanation](https://chatgpt.com/share/68ec8a8e-e474-800f-bce6-35612606fcf1)