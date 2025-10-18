import os
import joblib
import pandas as pd
from datetime import datetime

# MLflow & WandB
import mlflow
import mlflow.sklearn
import wandb

class Tracking:
    def __init__(self, backend="mlflow", project="ml_tracking", run_name=None, model=None):
        """
        backend: "mlflow" hoặc "wandb"
        project: tên experiment/run
        run_name: tên run cụ thể
        model: mô hình ML (tùy chọn)
        """
        self.backend = backend.lower()
        self.project = project
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = model
        self.active = None

        # Thư mục local
        self.model_dir = "experiments/models"
        self.data_dir = "data/processed"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    # ===== CONTEXT MANAGER =====
    def __enter__(self):
        if self.backend == "mlflow":
            mlflow.set_experiment(self.project)
            self.active = mlflow.start_run(run_name=self.run_name)
        elif self.backend == "wandb":
            self.active = wandb.init(project=self.project, name=self.run_name)
        else:
            raise ValueError("Chỉ hỗ trợ backend='mlflow' hoặc 'wandb'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.backend == "mlflow":
            mlflow.end_run()
        elif self.backend == "wandb":
            wandb.finish()

    # ===== LOG PARAMS =====
    def log_params(self, params: dict):
        # Flatten nested dict nếu có
        flat_params = self._flatten_dict(params)
        if self.backend == "mlflow":
            for k, v in flat_params.items():
                mlflow.log_param(k, v)
        elif self.backend == "wandb":
            wandb.config.update(flat_params)

    # ===== LOG METRICS =====
    def log_metrics(self, metrics: dict):
        flat_metrics = self._flatten_dict(metrics)
        if self.backend == "mlflow":
            for k, v in flat_metrics.items():
                mlflow.log_metric(k, v)
        elif self.backend == "wandb":
            wandb.log(flat_metrics)

    # ===== SAVE MODEL =====
    def save_model(self, model=None, filename=None):
        model = model or self.model
        if model is None:
            print("⚠️ Không có model để lưu.")
            return None

        # 1. Lưu local
        filename = filename or f"{self.run_name}_model.pkl"
        local_path = os.path.join(self.model_dir, filename)
        joblib.dump(model, local_path)
        print(f"✅ Mô hình đã lưu local: {local_path}")

        # 2. Log artifact
        if self.backend == "mlflow":
            # log sklearn model chuẩn MLflow
            mlflow.sklearn.log_model(model, artifact_path="model")
        elif self.backend == "wandb":
            # log WandB artifact
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(local_path)
            wandb.log_artifact(artifact)

        return local_path

    # ===== SAVE DATAFRAME =====
    def save_data(self, df: pd.DataFrame, filename=None):
        filename = filename or f"{self.run_name}_data.csv"
        file_path = os.path.join(self.data_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"✅ Dữ liệu đã lưu local: {file_path}")

        # Log artifact
        if self.backend == "mlflow":
            mlflow.log_artifact(file_path)
        elif self.backend == "wandb":
            artifact = wandb.Artifact('data', type='dataset')
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
        return file_path

    # ===== LOG ARTIFACT BẤT KỲ =====
    def log_artifact(self, file_path, artifact_name=None, artifact_type="dataset"):
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} không tồn tại.")
            return
        if self.backend == "mlflow":
            mlflow.log_artifact(file_path)
        elif self.backend == "wandb":
            artifact_name = artifact_name or os.path.basename(file_path)
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)

    # ====== PRIVATE: FLATTEN DICT =====
    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
