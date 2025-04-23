import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import json
import joblib
import time
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import config_manager, path_manager, constants
from app.config.utils import save_metadata, load_metadata, generate_model_version, cleanup_old_model_versions


class IncrementalModelTrainer:
    """
    Lớp huấn luyện tăng cường mô hình
    """

    def __init__(self, customer_id: str, version: str = None):
        """
        Khởi tạo IncrementalModelTrainer

        Args:
            customer_id: ID của khách hàng
            version: Phiên bản mô hình mới (nếu None, sẽ tạo mới)
        """
        self.customer_id = customer_id
        self.version = version or generate_model_version()

        # Tải cấu hình
        self.column_config = config_manager.get_column_config(customer_id)
        self.preprocess_config = config_manager.get_preprocessing_config(customer_id)
        self.model_config = config_manager.get_customer_config(customer_id).get('model_config', {})
        self.training_config = config_manager.get_training_config(customer_id)
        self.incremental_config = config_manager.get_customer_config(customer_id).get('incremental_training_config', {})

        # Lấy các tham số từ cấu hình
        self.vietnamese_text_columns = self.column_config.get('vietnamese_text_columns', [])
        self.id_columns = self.column_config.get('id_columns', [])
        self.optional_columns = self.column_config.get('optional_columns', [])

        # Cột mục tiêu
        self.primary_target = self.column_config.get('target_columns', {}).get('primary')
        self.secondary_target = self.column_config.get('target_columns', {}).get('secondary')
        self.condition_column = self.column_config.get('target_columns', {}).get('secondary_condition', {}).get(
            'column')
        self.starts_with = self.column_config.get('target_columns', {}).get('secondary_condition', {}).get(
            'starts_with')

        # Tải mô hình hiện có
        self.existing_hachtoan_model = None
        self.existing_mahanghua_model = None
        self.existing_outlier_model = None
        self.existing_label_encoders = None
        self.existing_metadata = None

        self._load_existing_models()

        # Các mô hình đã được cập nhật
        self.hachtoan_model = None
        self.mahanghua_model = None
        self.outlier_model = None
        self.label_encoders = None

        # Metadata
        self.metadata = {
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat(),
            "version": self.version,
            "is_incremental": True,
            "models": {}
        }

        # Nếu có metadata cũ, giữ lại một số thông tin
        if self.existing_metadata:
            self.metadata["previous_version"] = self.existing_metadata.get("version", "unknown")
            self.metadata["incremental_history"] = self.existing_metadata.get("incremental_history", [])
            self.metadata["incremental_history"].append({
                "from_version": self.existing_metadata.get("version", "unknown"),
                "to_version": self.version,
                "timestamp": datetime.now().isoformat()
            })

    def _load_existing_models(self):
        """Tải các mô hình hiện có từ đĩa"""
        try:
            # Xác định đường dẫn mô hình
            model_dir = path_manager.get_customer_model_path(self.customer_id)

            # Tải mô hình HachToan
            hachtoan_path = os.path.join(model_dir, 'hachtoan', 'model_latest.joblib')
            if os.path.exists(hachtoan_path):
                self.existing_hachtoan_model = joblib.load(hachtoan_path)
                logger.info(f"Đã tải mô hình HachToan hiện có từ: {hachtoan_path}")

            # Tải mô hình MaHangHoa
            mahanghua_path = os.path.join(model_dir, 'mahanghua', 'model_latest.joblib')
            if os.path.exists(mahanghua_path):
                self.existing_mahanghua_model = joblib.load(mahanghua_path)
                logger.info(f"Đã tải mô hình MaHangHoa hiện có từ: {mahanghua_path}")

            # Tải mô hình phát hiện outlier
            outlier_path = os.path.join(model_dir, 'outlier', 'model_latest.joblib')
            if os.path.exists(outlier_path):
                self.existing_outlier_model = joblib.load(outlier_path)
                logger.info(f"Đã tải mô hình phát hiện outlier hiện có từ: {outlier_path}")

            # Tải label encoders
            encoders_path = os.path.join(model_dir, 'encoders', 'label_encoders_latest.joblib')
            if os.path.exists(encoders_path):
                self.existing_label_encoders = joblib.load(encoders_path)
                logger.info(f"Đã tải label encoders hiện có từ: {encoders_path}")
                # Sao chép label encoders hiện có để sử dụng
                self.label_encoders = copy.deepcopy(self.existing_label_encoders)

            # Tải metadata
            metadata_path = os.path.join(model_dir, 'metadata', 'model_metadata_latest.json')
            if os.path.exists(metadata_path):
                self.existing_metadata = load_metadata(metadata_path)
                logger.info(f"Đã tải metadata hiện có từ: {metadata_path}")

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình hiện có: {str(e)}")
            logger.warning("Tiếp tục mà không có mô hình hiện có")

    def _extract_base_model(self, pipeline, model_type):
        """
        Trích xuất mô hình cơ sở từ pipeline

        Args:
            pipeline: Pipeline chứa mô hình
            model_type: Loại mô hình ('xgboost', 'lightgbm', etc.)

        Returns:
            Mô hình cơ sở
        """
        if pipeline is None:
            return None

        try:
            # Đối với XGBoost và LightGBM, mô hình cơ sở nằm trong bước 'classifier'
            if 'classifier' in pipeline.named_steps:
                base_model = pipeline.named_steps['classifier']
                logger.info(f"Đã trích xuất mô hình {model_type} cơ sở từ pipeline")
                return base_model
            else:
                logger.warning(f"Không tìm thấy bước 'classifier' trong pipeline")
                return None
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất mô hình cơ sở: {str(e)}")
            return None

    def _can_perform_incremental_training(self, model):
        """
        Kiểm tra xem mô hình có hỗ trợ huấn luyện tăng cường không

        Args:
            model: Mô hình cần kiểm tra

        Returns:
            True nếu hỗ trợ, False nếu không
        """
        if model is None:
            return False

        # Xác định loại mô hình
        model_type = type(model).__module__ + "." + type(model).__name__

        # XGBoost hỗ trợ huấn luyện tăng cường
        if 'xgboost' in model_type.lower():
            return True

        # LightGBM hỗ trợ huấn luyện tăng cường
        if 'lightgbm' in model_type.lower():
            return True

        # Các mô hình khác không hỗ trợ
        return False

    def _incremental_train_xgboost(self, base_model, X, y, model_config):
        """
        Huấn luyện tăng cường mô hình XGBoost

        Args:
            base_model: Mô hình XGBoost cơ sở
            X: Dữ liệu đặc trưng
            y: Nhãn
            model_config: Cấu hình mô hình

        Returns:
            Mô hình XGBoost đã được cập nhật
        """
        import xgboost as xgb

        # Tạo DMatrix từ dữ liệu mới
        dtrain = xgb.DMatrix(X, label=y)

        # Lấy tham số từ mô hình cơ sở
        params = base_model.get_params()

        # Điều chỉnh số vòng lặp
        num_boost_round = params.pop('n_estimators', 10)  # Số vòng lặp mặc định
        incremental_rounds = self.incremental_config.get('num_boost_round', num_boost_round // 2)

        # Điều chỉnh learning rate cho huấn luyện tăng cường
        incremental_lr = self.incremental_config.get('learning_rate', params.get('learning_rate', 0.1))
        params['learning_rate'] = incremental_lr

        # Tùy chỉnh các tham số khác
        params.pop('callbacks', None)  # Loại bỏ callbacks từ get_params()

        # Lấy mô hình Booster từ mô hình scikit-learn
        xgb_model = base_model.get_booster()

        # Huấn luyện thêm trên dữ liệu mới
        logger.info(f"Huấn luyện tăng cường XGBoost với {incremental_rounds} vòng lặp, learning_rate={incremental_lr}")

        updated_booster = xgb.train(
            params,
            dtrain,
            num_boost_round=incremental_rounds,
            xgb_model=xgb_model
        )

        # Tạo mô hình scikit-learn mới với Booster đã cập nhật
        updated_model = xgb.XGBClassifier(**params)
        updated_model._Booster = updated_booster

        return updated_model

    def _incremental_train_lightgbm(self, base_model, X, y, model_config):
        """
        Huấn luyện tăng cường mô hình LightGBM

        Args:
            base_model: Mô hình LightGBM cơ sở
            X: Dữ liệu đặc trưng
            y: Nhãn
            model_config: Cấu hình mô hình

        Returns:
            Mô hình LightGBM đã được cập nhật
        """
        import lightgbm as lgb

        # Tạo Dataset từ dữ liệu mới
        dtrain = lgb.Dataset(X, label=y)

        # Lấy tham số từ mô hình cơ sở
        params = base_model.get_params()

        # Điều chỉnh số vòng lặp
        num_boost_round = params.pop('n_estimators', 10)  # Số vòng lặp mặc định
        incremental_rounds = self.incremental_config.get('num_boost_round', num_boost_round // 2)

        # Điều chỉnh learning rate cho huấn luyện tăng cường
        incremental_lr = self.incremental_config.get('learning_rate', params.get('learning_rate', 0.1))
        params['learning_rate'] = incremental_lr

        # Tùy chỉnh các tham số khác
        params.pop('callbacks', None)  # Loại bỏ callbacks từ get_params()

        # Lấy mô hình Booster từ mô hình scikit-learn
        lgb_model = base_model.booster_

        # Huấn luyện thêm trên dữ liệu mới
        logger.info(f"Huấn luyện tăng cường LightGBM với {incremental_rounds} vòng lặp, learning_rate={incremental_lr}")

        updated_booster = lgb.train(
            params,
            dtrain,
            num_boost_round=incremental_rounds,
            init_model=lgb_model
        )

        # Tạo mô hình scikit-learn mới với Booster đã cập nhật
        updated_model = lgb.LGBMClassifier(**params)
        updated_model._Booster = updated_booster

        return updated_model

    def _update_pipeline_with_model(self, original_pipeline, updated_model):
        """
        Cập nhật pipeline với mô hình đã được huấn luyện tăng cường

        Args:
            original_pipeline: Pipeline gốc
            updated_model: Mô hình đã được cập nhật

        Returns:
            Pipeline đã được cập nhật
        """
        if original_pipeline is None or updated_model is None:
            return None

        # Tạo bản sao của pipeline gốc
        updated_pipeline = copy.deepcopy(original_pipeline)

        # Thay thế mô hình trong pipeline
        if 'classifier' in updated_pipeline.named_steps:
            updated_pipeline.named_steps['classifier'] = updated_model
            logger.info("Đã cập nhật mô hình trong pipeline")
            return updated_pipeline
        else:
            logger.warning("Không tìm thấy bước 'classifier' trong pipeline")
            return original_pipeline

    def incremental_train_hachtoan_model(self, df: pd.DataFrame):
        """
        Huấn luyện tăng cường mô hình dự đoán HachToan

        Args:
            df: DataFrame chứa dữ liệu huấn luyện mới
        """
        logger.info(f"Bắt đầu huấn luyện tăng cường mô hình HachToan cho khách hàng {self.customer_id}")

        # Kiểm tra mô hình hiện có
        if self.existing_hachtoan_model is None:
            logger.warning("Không tìm thấy mô hình HachToan hiện có, thực hiện huấn luyện mới")
            trainer = ModelTrainer(self.customer_id)
            trainer.train_hachtoan_model(df)
            self.hachtoan_model = trainer.hachtoan_model
            if self.label_encoders is None:
                self.label_encoders = {}
            if hasattr(trainer, 'label_encoders') and trainer.label_encoders:
                self.label_encoders.update(trainer.label_encoders)
            return

        # Chuẩn bị đặc trưng và nhãn
        X = df.drop(columns=[self.primary_target, self.secondary_target], errors='ignore')
        y = df[self.primary_target]

        # Mã hóa nhãn nếu cần
        if 'hachtoan' in self.label_encoders:
            try:
                y = self.label_encoders['hachtoan'].transform(y)
            except:
                # Nếu có nhãn mới, cần cập nhật encoder
                logger.warning("Phát hiện nhãn HachToan mới, cập nhật label encoder")
                old_classes = set(self.label_encoders['hachtoan'].classes_)
                new_classes = set(y.unique())
                added_classes = new_classes - old_classes
                logger.info(f"Các nhãn mới: {added_classes}")

                # Tạo encoder mới
                from sklearn.preprocessing import LabelEncoder
                new_encoder = LabelEncoder()
                new_encoder.fit(list(old_classes) + list(added_classes))

                # Lưu lại các lớp cũ để có thể ánh xạ giữa hai encoder
                old_encoder = self.label_encoders['hachtoan']

                # Cập nhật encoder
                self.label_encoders['hachtoan'] = new_encoder
                y = new_encoder.transform(y)
        elif not y.dtype.name == 'int64' and not y.dtype.name == 'int32':
            logger.info(f"Mã hóa nhãn HachToan từ kiểu {y.dtype}")
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            if self.label_encoders is None:
                self.label_encoders = {}
            self.label_encoders['hachtoan'] = encoder

        # Trích xuất mô hình cơ sở từ pipeline
        base_model = self._extract_base_model(self.existing_hachtoan_model, 'hachtoan_model')

        # Kiểm tra khả năng huấn luyện tăng cường
        if base_model is not None and self._can_perform_incremental_training(base_model):
            try:
                # Xác định loại mô hình và huấn luyện tăng cường
                model_type = type(base_model).__module__ + "." + type(base_model).__name__

                # Tiền xử lý dữ liệu sử dụng transformer từ pipeline gốc
                if 'features' in self.existing_hachtoan_model.named_steps:
                    transformer = self.existing_hachtoan_model.named_steps['features']
                    X_transformed = transformer.transform(X)
                else:
                    logger.warning("Không tìm thấy bước 'features' trong pipeline, sử dụng dữ liệu thô")
                    X_transformed = X

                # Huấn luyện tăng cường tùy theo loại mô hình
                if 'xgboost' in model_type.lower():
                    logger.info("Thực hiện huấn luyện tăng cường XGBoost cho mô hình HachToan")
                    updated_model = self._incremental_train_xgboost(
                        base_model, X_transformed, y, self.model_config.get('hachtoan_model', {})
                    )
                elif 'lightgbm' in model_type.lower():
                    logger.info("Thực hiện huấn luyện tăng cường LightGBM cho mô hình HachToan")
                    updated_model = self._incremental_train_lightgbm(
                        base_model, X_transformed, y, self.model_config.get('hachtoan_model', {})
                    )
                else:
                    logger.warning(f"Không hỗ trợ huấn luyện tăng cường cho mô hình loại: {model_type}")
                    # Huấn luyện mới nếu không hỗ trợ huấn luyện tăng cường
                    trainer = ModelTrainer(self.customer_id)
                    trainer.train_mahanghua_model(df)
                    self.mahanghua_model = trainer.mahanghua_model
                    return

                # Cập nhật pipeline với mô hình mới
                self.mahanghua_model = self._update_pipeline_with_model(
                    self.existing_mahanghua_model, updated_model
                )

                # Thêm metadata
                self.metadata['models']['mahanghua'] = {
                    'incremental_training': True,
                    'base_version': self.existing_metadata.get('version', 'unknown'),
                    'training_time': None,  # Sẽ được cập nhật sau
                    'n_samples': len(X),
                    'algorithm': model_type,
                    'previous_n_estimators': getattr(base_model, 'n_estimators', None),
                    'added_n_estimators': self.incremental_config.get('num_boost_round',
                                                                      getattr(base_model, 'n_estimators', 10) // 2),
                    'condition': f"{self.condition_column}.startswith('{self.starts_with}')"
                    if self.condition_column and self.starts_with else None
                }

            except Exception as e:
                logger.exception(f"Lỗi khi huấn luyện tăng cường mô hình MaHangHoa: {str(e)}")
                # Huấn luyện mới nếu có lỗi
                logger.warning("Thực hiện huấn luyện mới do lỗi huấn luyện tăng cường")
                trainer = ModelTrainer(self.customer_id)
                trainer.train_mahanghua_model(df)
                self.mahanghua_model = trainer.mahanghua_model
        else:
            # Huấn luyện mới nếu không hỗ trợ huấn luyện tăng cường
            logger.info("Mô hình hiện có không hỗ trợ huấn luyện tăng cường, thực hiện huấn luyện mới")
            trainer = ModelTrainer(self.customer_id)
            trainer.train_mahanghua_model(df)
            self.mahanghua_model = trainer.mahanghua_model
            if self.label_encoders is None:
                self.label_encoders = {}
            if hasattr(trainer, 'label_encoders') and trainer.label_encoders:
                self.label_encoders.update(trainer.label_encoders)
                ếu
                không
                hỗ
                trợ
                huấn
                luyện
                tăng
                cường
                trainer = ModelTrainer(self.customer_id)
                trainer.train_hachtoan_model(df)
                self.hachtoan_model = trainer.hachtoan_model
                return

            # Cập nhật pipeline với mô hình mới
            self.hachtoan_model = self._update_pipeline_with_model(
                self.existing_hachtoan_model, updated_model
            )

            # Thêm metadata
            self.metadata['models']['hachtoan'] = {
                'incremental_training': True,
                'base_version': self.existing_metadata.get('version', 'unknown'),
                'training_time': None,  # Sẽ được cập nhật sau
                'n_samples': len(X),
                'algorithm': model_type,
                'previous_n_estimators': getattr(base_model, 'n_estimators', None),
                'added_n_estimators': self.incremental_config.get('num_boost_round',
                                                                  getattr(base_model, 'n_estimators', 10) // 2)
            }

        except Exception as e:
        logger.exception(f"Lỗi khi huấn luyện tăng cường mô hình HachToan: {str(e)}")
        # Huấn luyện mới nếu có lỗi
        logger.warning("Thực hiện huấn luyện mới do lỗi huấn luyện tăng cường")
        trainer = ModelTrainer(self.customer_id)
        trainer.train_hachtoan_model(df)
        self.hachtoan_model = trainer.hachtoan_model

else:
# Huấn luyện mới nếu không hỗ trợ huấn luyện tăng cường
logger.info("Mô hình hiện có không hỗ trợ huấn luyện tăng cường, thực hiện huấn luyện mới")
trainer = ModelTrainer(self.customer_id)
trainer.train_hachtoan_model(df)
self.hachtoan_model = trainer.hachtoan_model
if self.label_encoders is None:
    self.label_encoders = {}
if hasattr(trainer, 'label_encoders') and trainer.label_encoders:
    self.label_encoders.update(trainer.label_encoders)


def incremental_train_mahanghua_model(self, df: pd.DataFrame):
    """
    Huấn luyện tăng cường mô hình dự đoán MaHangHoa

    Args:
        df: DataFrame chứa dữ liệu huấn luyện mới
    """
    logger.info(f"Bắt đầu huấn luyện tăng cường mô hình MaHangHoa cho khách hàng {self.customer_id}")

    # Lọc dữ liệu theo điều kiện
    if self.condition_column and self.starts_with:
        condition_mask = df[self.condition_column].astype(str).str.startswith(self.starts_with)
        if condition_mask.sum() == 0:
            logger.warning(
                f"Không có dữ liệu nào thỏa điều kiện {self.condition_column}.startswith('{self.starts_with}')"
            )
            logger.warning("Bỏ qua huấn luyện mô hình MaHangHoa")
            return

        logger.info(f"Lọc dữ liệu theo điều kiện: {condition_mask.sum()}/{len(df)} dòng thỏa mãn")
        filtered_df = df[condition_mask].reset_index(drop=True)
    else:
        # Nếu không có điều kiện, sử dụng toàn bộ dữ liệu
        filtered_df = df

    # Kiểm tra xem có đủ dữ liệu để huấn luyện không
    if len(filtered_df) < 10:
        logger.warning(f"Không đủ dữ liệu để huấn luyện mô hình MaHangHoa: chỉ có {len(filtered_df)} dòng")
        return

    # Kiểm tra mô hình hiện có
    if self.existing_mahanghua_model is None:
        logger.warning("Không tìm thấy mô hình MaHangHoa hiện có, thực hiện huấn luyện mới")
        trainer = ModelTrainer(self.customer_id)
        trainer.train_mahanghua_model(df)
        self.mahanghua_model = trainer.mahanghua_model
        if self.label_encoders is None:
            self.label_encoders = {}
        if hasattr(trainer, 'label_encoders') and trainer.label_encoders:
            self.label_encoders.update(trainer.label_encoders)
        return

    # Chuẩn bị đặc trưng và nhãn
    X = filtered_df.drop(columns=[self.secondary_target], errors='ignore')
    y = filtered_df[self.secondary_target]

    # Mã hóa nhãn nếu cần
    if 'mahanghua' in self.label_encoders:
        try:
            y = self.label_encoders['mahanghua'].transform(y)
        except:
            # Nếu có nhãn mới, cần cập nhật encoder
            logger.warning("Phát hiện nhãn MaHangHoa mới, cập nhật label encoder")
            old_classes = set(self.label_encoders['mahanghua'].classes_)
            new_classes = set(y.unique())
            added_classes = new_classes - old_classes
            logger.info(f"Các nhãn mới: {added_classes}")

            # Tạo encoder mới
            from sklearn.preprocessing import LabelEncoder
            new_encoder = LabelEncoder()
            new_encoder.fit(list(old_classes) + list(added_classes))

            # Lưu lại các lớp cũ để có thể ánh xạ giữa hai encoder
            old_encoder = self.label_encoders['mahanghua']

            # Cập nhật encoder
            self.label_encoders['mahanghua'] = new_encoder
            y = new_encoder.transform(y)
    elif not y.dtype.name == 'int64' and not y.dtype.name == 'int32':
        logger.info(f"Mã hóa nhãn MaHangHoa từ kiểu {y.dtype}")
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        if self.label_encoders is None:
            self.label_encoders = {}
        self.label_encoders['mahanghua'] = encoder

    # Trích xuất mô hình cơ sở từ pipeline
    base_model = self._extract_base_model(self.existing_mahanghua_model, 'mahanghua_model')

    # Kiểm tra khả năng huấn luyện tăng cường
    if base_model is not None and self._can_perform_incremental_training(base_model):
        try:
            # Xác định loại mô hình và huấn luyện tăng cường
            model_type = type(base_model).__module__ + "." + type(base_model).__name__

            # Tiền xử lý dữ liệu sử dụng transformer từ pipeline gốc
            if 'features' in self.existing_mahanghua_model.named_steps:
                transformer = self.existing_mahanghua_model.named_steps['features']
                X_transformed = transformer.transform(X)
            else:
                logger.warning("Không tìm thấy bước 'features' trong pipeline, sử dụng dữ liệu thô")
                X_transformed = X

            # Huấn luyện tăng cường tùy theo loại mô hình
            if 'xgboost' in model_type.lower():
                logger.info("Thực hiện huấn luyện tăng cường XGBoost cho mô hình MaHangHoa")
                updated_model = self._incremental_train_xgboost(
                    base_model, X_transformed, y, self.model_config.get('mahanghua_model', {})
                )
            elif 'lightgbm' in model_type.lower():
                logger.info("Thực hiện huấn luyện tăng cường LightGBM cho mô hình MaHangHoa")
                updated_model = self._incremental_train_lightgbm(
                    base_model, X_transformed, y, self.model_config.get('mahanghua_model', {})
                )
                else:
                logger.warning(f"Không hỗ trợ huấn luyện tăng cường cho mô hình loại: {model_type}")
                # Huấn luyện mới nếu không hỗ trợ huấn luyện tăng cường
                trainer = ModelTrainer(self.customer_id)
                trainer.train_mahanghua_model(df)
                self.mahanghua_model = trainer.mahanghua_model
                return

                # Cập nhật pipeline với mô hình mới
            self.mahanghua_model = self._update_pipeline_with_model(
                self.existing_mahanghua_model, updated_model
            )

            # Thêm metadata
            self.metadata['models']['mahanghua'] = {
                'incremental_training': True,
                'base_version': self.existing_metadata.get('version', 'unknown'),
                'training_time': None,  # Sẽ được cập nhật sau
                'n_samples': len(X),
                'algorithm': model_type,
                'previous_n_estimators': getattr(base_model, 'n_estimators', None),
                'added_n_estimators': self.incremental_config.get('num_boost_round',
                                                                  getattr(base_model, 'n_estimators', 10) // 2),
                'condition': f"{self.condition_column}.startswith('{self.starts_with}')"
                if self.condition_column and self.starts_with else None
            }

        except Exception as e:
            logger.exception(f"Lỗi khi huấn luyện tăng cường mô hình MaHangHoa: {str(e)}")
            # Huấn luyện mới nếu có lỗi
            logger.warning("Thực hiện huấn luyện mới do lỗi huấn luyện tăng cường")
            trainer = ModelTrainer(self.customer_id)
            trainer.train_mahanghua_model(df)
            self.mahanghua_model = trainer.mahanghua_model
        else:
            # Huấn luyện mới nếu không hỗ trợ huấn luyện tăng cường
            logger.info("Mô hình hiện có không hỗ trợ huấn luyện tăng cường, thực hiện huấn luyện mới")
            trainer = ModelTrainer(self.customer_id)
            trainer.train_mahanghua_model(df)
            self.mahanghua_model = trainer.mahanghua_model
            if self.label_encoders is None:
                self.label_encoders = {}
            if hasattr(trainer, 'label_encoders') and trainer.label_encoders:
                self.label_encoders.update(trainer.label_encoders)