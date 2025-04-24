#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script huấn luyện mô hình
- Huấn luyện mô hình phân loại HachToan
- Huấn luyện mô hình phân loại MaHangHoa
- Phát hiện outlier
"""

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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import config_manager, path_manager, constants
from app.config.utils import save_metadata, load_metadata, generate_model_version, cleanup_old_model_versions
from app.utils.feature_extraction import FeatureExtractor
from app.utils.evaluation import evaluate_model, cross_validate_model, compute_confusion_matrix

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='train.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('train')

class ModelTrainer:
    """
    Lớp huấn luyện mô hình
    - Huấn luyện mô hình phân loại HachToan
    - Huấn luyện mô hình phân loại MaHangHoa
    - Phát hiện outlier
    """

    def __init__(self, customer_id: str, version: str = None):
        """
        Khởi tạo ModelTrainer

        Args:
            customer_id: ID của khách hàng
            version: Phiên bản mô hình (nếu None, sẽ tạo mới)
        """
        self.customer_id = customer_id
        self.version = version or generate_model_version()

        # Tải cấu hình
        self.column_config = config_manager.get_column_config(customer_id)
        self.preprocess_config = config_manager.get_preprocessing_config(customer_id)
        self.model_config = config_manager.get_model_config(customer_id, 'hachtoan_model')
        self.training_config = config_manager.get_training_config(customer_id)

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

        # Feature Extractor
        self.feature_extractor = None

        # Label encoders
        self.label_encoders = {}

        # Mô hình
        self.hachtoan_model = None
        self.mahanghua_model = None
        self.outlier_model = None

        # Metadata
        self.metadata = {
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat(),
            "version": self.version,
            "is_incremental": False,
            "models": {}
        }

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu cho huấn luyện

        Args:
            df: DataFrame chứa dữ liệu

        Returns:
            DataFrame đã được tiền xử lý
        """
        logger.info("Tiền xử lý dữ liệu...")

        # Tạo một bản sao để tránh thay đổi dữ liệu gốc
        processed_df = df.copy()

        # Xử lý các cột văn bản tiếng Việt
        for column in self.vietnamese_text_columns:
            if column in processed_df.columns:
                # Điền các giá trị null
                processed_df[column] = processed_df[column].fillna("")

                # Chuẩn hóa văn bản nếu được yêu cầu
                if self.preprocess_config.get('normalize_text', True):
                    from app.config.utils import normalize_vietnamese_text
                    processed_df[column] = processed_df[column].apply(normalize_vietnamese_text)

        # Xử lý các cột ID
        for column in self.id_columns:
            if column in processed_df.columns:
                # Chuyển đổi thành chuỗi
                processed_df[column] = processed_df[column].astype(str)

                # Loại bỏ khoảng trắng thừa
                processed_df[column] = processed_df[column].str.strip()

                # Thay thế giá trị null
                processed_df[column] = processed_df[column].replace('nan', 'unknown')
                processed_df[column] = processed_df[column].fillna('unknown')

        return processed_df

    def _create_feature_extractor(self) -> FeatureExtractor:
        """
        Tạo feature extractor

        Returns:
            FeatureExtractor được cấu hình
        """
        logger.info("Tạo feature extractor...")

        # Xác định các cột đặc trưng
        text_columns = self.vietnamese_text_columns
        id_columns = self.id_columns
        # Thêm các cột khác nếu cần

        # Tạo feature extractor
        feature_extractor = FeatureExtractor(
            text_columns=text_columns,
            categorical_columns=id_columns,
            config=self.preprocess_config
        )

        return feature_extractor

    def _handle_class_imbalance(self, X: np.ndarray, y: np.ndarray, target_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Xử lý mất cân bằng dữ liệu

        Args:
            X: Ma trận đặc trưng
            y: Mảng nhãn
            target_type: Loại nhãn ('hachtoan' hoặc 'mahanghua')

        Returns:
            Tuple (X_resampled, y_resampled)
        """
        # Lấy cấu hình mô hình tương ứng
        model_config = config_manager.get_model_config(self.customer_id, f'{target_type}_model')

        # Kiểm tra xem có cần xử lý mất cân bằng hay không
        if model_config.get('handle_imbalance', True):
            strategy = model_config.get('imbalance_strategy', 'auto')
            logger.info(f"Xử lý mất cân bằng dữ liệu với chiến lược: {strategy}")

            # Đếm số lượng mẫu cho mỗi lớp
            class_counts = np.bincount(y)
            min_samples = min(class_counts)
            max_samples = max(class_counts)
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')

            # Nếu không mất cân bằng, không cần xử lý
            if imbalance_ratio <= constants.IMBALANCE_THRESHOLD:
                logger.info(
                    f"Tỷ lệ mất cân bằng {imbalance_ratio:.2f} <= {constants.IMBALANCE_THRESHOLD}, không cần xử lý")
                return X, y

            # Khởi tạo resampler tùy theo chiến lược
            if strategy == 'auto':
                # Tự động chọn chiến lược dựa trên mức độ mất cân bằng
                if imbalance_ratio > constants.SEVERE_IMBALANCE_THRESHOLD:
                    if len(class_counts) > 2:  # Nhiều lớp
                        strategy = 'smote'
                    else:  # Nhị phân
                        strategy = 'smoteenn'
                else:
                    strategy = 'class_weight'

                logger.info(f"Đã tự động chọn chiến lược: {strategy}")

            # Xử lý theo chiến lược đã chọn
            if strategy == 'class_weight':
                # Không cần resampling, sẽ sử dụng class_weight trong mô hình
                logger.info("Sẽ sử dụng class_weight trong mô hình thay vì resampling")
                return X, y

            try:
                # Import các thư viện cần thiết
                from imblearn.over_sampling import SMOTE, ADASYN
                from imblearn.under_sampling import RandomUnderSampler, NearMiss
                from imblearn.combine import SMOTEENN, SMOTETomek

                # Khởi tạo resampler tương ứng
                if strategy == 'smote':
                    resampler = SMOTE(random_state=constants.DEFAULT_RANDOM_STATE)
                elif strategy == 'adasyn':
                    resampler = ADASYN(random_state=constants.DEFAULT_RANDOM_STATE)
                elif strategy == 'smoteenn':
                    resampler = SMOTEENN(random_state=constants.DEFAULT_RANDOM_STATE)
                elif strategy == 'smotetomek':
                    resampler = SMOTETomek(random_state=constants.DEFAULT_RANDOM_STATE)
                elif strategy == 'undersampling':
                    resampler = RandomUnderSampler(random_state=constants.DEFAULT_RANDOM_STATE)
                elif strategy == 'nearmiss':
                    resampler = NearMiss()
                else:
                    logger.warning(f"Chiến lược {strategy} không được hỗ trợ, sử dụng SMOTE")
                    resampler = SMOTE(random_state=constants.DEFAULT_RANDOM_STATE)

                # Thực hiện resampling
                X_resampled, y_resampled = resampler.fit_resample(X, y)

                logger.info(f"Đã xử lý mất cân bằng: {len(y)} -> {len(y_resampled)} mẫu")
                return X_resampled, y_resampled

            except Exception as e:
                logger.error(f"Lỗi khi xử lý mất cân bằng: {str(e)}")
                logger.warning("Sử dụng dữ liệu gốc mà không xử lý mất cân bằng")
                return X, y
        else:
            logger.info("Không xử lý mất cân bằng dữ liệu theo cấu hình")
            return X, y

    def _create_model(self, model_type: str) -> Any:
        """
        Tạo mô hình phân loại

        Args:
            model_type: Loại mô hình ('hachtoan_model' hoặc 'mahanghua_model')

        Returns:
            Mô hình đã được cấu hình
        """
        # Lấy cấu hình mô hình
        model_config = config_manager.get_model_config(self.customer_id, model_type)
        model_type_name = model_config.get('type', 'xgboost').lower()
        params = model_config.get('params', {})

        logger.info(f"Tạo mô hình {model_type_name} cho {model_type}")

        # Tạo mô hình tương ứng
        try:
            if model_type_name == 'xgboost':
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**params)
                    logger.info(f"Đã tạo mô hình XGBoost với tham số: {params}")
                    return model
                except ImportError:
                    logger.error("Không thể import thư viện xgboost, sử dụng RandomForest thay thế")
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(n_estimators=100, random_state=constants.DEFAULT_RANDOM_STATE)

            elif model_type_name == 'lightgbm':
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**params)
                    logger.info(f"Đã tạo mô hình LightGBM với tham số: {params}")
                    return model
                except ImportError:
                    logger.error("Không thể import thư viện lightgbm, sử dụng RandomForest thay thế")
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(n_estimators=100, random_state=constants.DEFAULT_RANDOM_STATE)

            elif model_type_name == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                n_estimators = params.get('n_estimators', 100)
                max_depth = params.get('max_depth', None)
                random_state = params.get('random_state', constants.DEFAULT_RANDOM_STATE)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               random_state=random_state)
                logger.info(f"Đã tạo mô hình RandomForest với tham số: {params}")
                return model

            elif model_type_name == 'svm':
                from sklearn.svm import SVC
                model = SVC(probability=True, **params)
                logger.info(f"Đã tạo mô hình SVM với tham số: {params}")
                return model

            else:
                logger.warning(f"Loại mô hình không hỗ trợ: {model_type_name}, sử dụng RandomForest thay thế")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=constants.DEFAULT_RANDOM_STATE)

        except Exception as e:
            logger.error(f"Lỗi khi tạo mô hình {model_type_name}: {str(e)}")
            logger.warning("Sử dụng RandomForest thay thế")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=constants.DEFAULT_RANDOM_STATE)

        def _create_outlier_detector(self) -> Any:
            """
            Tạo mô hình phát hiện outlier

            Returns:
                Mô hình phát hiện outlier
            """
            # Lấy cấu hình
            outlier_config = self.model_config.get('outlier_detection', {})
            enabled = outlier_config.get('enabled', True)

            if not enabled:
                logger.info("Phát hiện outlier bị tắt trong cấu hình")
                return None

            method = outlier_config.get('method', 'isolation_forest')
            params = outlier_config.get('params', {})

            logger.info(f"Tạo mô hình phát hiện outlier {method}")

            try:
                if method == 'isolation_forest':
                    from sklearn.ensemble import IsolationForest
                    contamination = params.get('contamination', 'auto')
                    n_estimators = params.get('n_estimators', 100)
                    random_state = params.get('random_state', constants.DEFAULT_RANDOM_STATE)
                    model = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                                            random_state=random_state)
                    return model

                elif method == 'one_class_svm':
                    from sklearn.svm import OneClassSVM
                    nu = params.get('nu', 0.1)
                    model = OneClassSVM(nu=nu)
                    return model

                elif method == 'local_outlier_factor':
                    from sklearn.neighbors import LocalOutlierFactor
                    n_neighbors = params.get('n_neighbors', 20)
                    contamination = params.get('contamination', 'auto')
                    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
                    return model

                elif method == 'elliptic_envelope':
                    from sklearn.covariance import EllipticEnvelope
                    contamination = params.get('contamination', 0.1)
                    random_state = params.get('random_state', constants.DEFAULT_RANDOM_STATE)
                    model = EllipticEnvelope(contamination=contamination, random_state=random_state)
                    return model

                else:
                    logger.warning(
                        f"Phương pháp phát hiện outlier không hỗ trợ: {method}, sử dụng IsolationForest thay thế")
                    from sklearn.ensemble import IsolationForest
                    return IsolationForest(random_state=constants.DEFAULT_RANDOM_STATE)

            except Exception as e:
                logger.error(f"Lỗi khi tạo mô hình phát hiện outlier {method}: {str(e)}")
                logger.warning("Sử dụng IsolationForest thay thế")
                from sklearn.ensemble import IsolationForest
                return IsolationForest(random_state=constants.DEFAULT_RANDOM_STATE)

        def _create_pipeline(self, feature_extractor, classifier):
            """
            Tạo pipeline kết hợp feature extractor và classifier

            Args:
                feature_extractor: FeatureExtractor
                classifier: Mô hình phân loại

            Returns:
                Pipeline
            """
            from sklearn.pipeline import Pipeline

            logger.info("Tạo pipeline kết hợp feature extractor và classifier")

            steps = [
                ('features', feature_extractor),
                ('classifier', classifier)
            ]

            return Pipeline(steps)

        def train_hachtoan_model(self, df: pd.DataFrame) -> None:
            """
            Huấn luyện mô hình dự đoán HachToan

            Args:
                df: DataFrame chứa dữ liệu huấn luyện
            """
            logger.info(f"Bắt đầu huấn luyện mô hình HachToan cho khách hàng {self.customer_id}")
            start_time = time.time()

            # Kiểm tra dữ liệu
            if self.primary_target not in df.columns:
                raise ValueError(f"Không tìm thấy cột mục tiêu {self.primary_target}")

            # Tiền xử lý dữ liệu
            df_processed = self._preprocess_data(df)

            # Tạo feature extractor
            if self.feature_extractor is None:
                self.feature_extractor = self._create_feature_extractor()

            # Chuẩn bị đặc trưng và nhãn
            X = df_processed.drop(columns=[self.primary_target, self.secondary_target], errors='ignore')
            y = df_processed[self.primary_target]

            # Mã hóa nhãn nếu cần
            if not y.dtype.name == 'int64' and not y.dtype.name == 'int32':
                logger.info(f"Mã hóa nhãn HachToan từ kiểu {y.dtype}")
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                self.label_encoders['hachtoan'] = encoder

            # Trích xuất đặc trưng
            logger.info("Trích xuất đặc trưng...")
            X_transformed = self.feature_extractor.fit_transform(X)
            logger.info(f"Đã trích xuất {X_transformed.shape[1]} đặc trưng")

            # Xử lý mất cân bằng dữ liệu
            X_balanced, y_balanced = self._handle_class_imbalance(X_transformed, y, 'hachtoan')

            # Tạo mô hình
            classifier = self._create_model('hachtoan_model')

            # Cấu hình class_weight nếu cần
            model_config = config_manager.get_model_config(self.customer_id, 'hachtoan_model')
            if model_config.get('handle_imbalance', True) and model_config.get('imbalance_strategy',
                                                                               'auto') == 'class_weight':
                if hasattr(classifier, 'set_params'):
                    try:
                        logger.info("Cấu hình class_weight='balanced'")
                        classifier.set_params(class_weight='balanced')
                    except:
                        logger.warning("Không thể đặt class_weight='balanced'")

            # Huấn luyện mô hình
            logger.info("Huấn luyện mô hình...")
            classifier.fit(X_balanced, y_balanced)

            # Tạo pipeline để dự đoán
            self.hachtoan_model = self._create_pipeline(self.feature_extractor, classifier)

            # Thời gian huấn luyện
            training_time = time.time() - start_time
            logger.info(f"Hoàn tất huấn luyện trong {training_time:.2f} giây")

            # Thêm metadata
            self.metadata['models']['hachtoan'] = {
                'training_time': training_time,
                'n_samples': len(df),
                'n_features': X_transformed.shape[1],
                'n_classes': len(np.unique(y)),
                'feature_importance': None,
                'class_distribution': {str(i): int(count) for i, count in enumerate(np.bincount(y))}
            }

            # Lưu feature importance nếu có
            if hasattr(classifier, 'feature_importances_'):
                try:
                    feature_names = self.feature_extractor.get_feature_names()
                    top_features = []

                    for i, importance in enumerate(classifier.feature_importances_):
                        if i < len(feature_names):
                            top_features.append({
                                'feature': feature_names[i],
                                'importance': float(importance)
                            })

                    # Sắp xếp theo độ quan trọng giảm dần và lấy top 20
                    top_features.sort(key=lambda x: x['importance'], reverse=True)
                    top_features = top_features[:20]

                    self.metadata['models']['hachtoan']['feature_importance'] = top_features
                except Exception as e:
                    logger.error(f"Lỗi khi lưu feature importance: {str(e)}")

        def train_mahanghua_model(self, df: pd.DataFrame) -> None:
            """
            Huấn luyện mô hình dự đoán MaHangHoa

            Args:
                df: DataFrame chứa dữ liệu huấn luyện
            """
            logger.info(f"Bắt đầu huấn luyện mô hình MaHangHoa cho khách hàng {self.customer_id}")

            # Kiểm tra dữ liệu
            if self.secondary_target not in df.columns:
                logger.warning(f"Không tìm thấy cột mục tiêu {self.secondary_target}")
                return

            # Lọc dữ liệu theo điều kiện
            if self.condition_column and self.starts_with:
                condition_mask = df[self.condition_column].astype(str).str.startswith(self.starts_with)
                if condition_mask.sum() == 0:
                    logger.warning(
                        f"Không có dữ liệu nào thỏa điều kiện {self.condition_column}.startswith('{self.starts_with}')")
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

            start_time = time.time()

            # Tiền xử lý dữ liệu
            df_processed = self._preprocess_data(filtered_df)

            # Tạo feature extractor nếu chưa có
            if self.feature_extractor is None:
                self.feature_extractor = self._create_feature_extractor()

            # Chuẩn bị đặc trưng và nhãn
            X = df_processed.drop(columns=[self.secondary_target], errors='ignore')
            y = df_processed[self.secondary_target]

            # Mã hóa nhãn nếu cần
            if not y.dtype.name == 'int64' and not y.dtype.name == 'int32':
                logger.info(f"Mã hóa nhãn MaHangHoa từ kiểu {y.dtype}")
                from sklearn.preprocessing import LabelEncoder
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                self.label_encoders['mahanghua'] = encoder

            # Trích xuất đặc trưng
            logger.info("Trích xuất đặc trưng...")
            X_transformed = self.feature_extractor.fit_transform(X)
            logger.info(f"Đã trích xuất {X_transformed.shape[1]} đặc trưng")

            # Xử lý mất cân bằng dữ liệu
            X_balanced, y_balanced = self._handle_class_imbalance(X_transformed, y, 'mahanghua')

            # Tạo mô hình
            classifier = self._create_model('mahanghua_model')

            # Cấu hình class_weight nếu cần
            model_config = config_manager.get_model_config(self.customer_id, 'mahanghua_model')
            if model_config.get('handle_imbalance', True) and model_config.get('imbalance_strategy',
                                                                               'auto') == 'class_weight':
                if hasattr(classifier, 'set_params'):
                    try:
                        logger.info("Cấu hình class_weight='balanced'")
                        classifier.set_params(class_weight='balanced')
                    except:
                        logger.warning("Không thể đặt class_weight='balanced'")

            # Huấn luyện mô hình
            logger.info("Huấn luyện mô hình...")
            classifier.fit(X_balanced, y_balanced)

            # Tạo pipeline để dự đoán
            self.mahanghua_model = self._create_pipeline(self.feature_extractor, classifier)

            # Thời gian huấn luyện
            training_time = time.time() - start_time
            logger.info(f"Hoàn tất huấn luyện trong {training_time:.2f} giây")

            # Thêm metadata
            self.metadata['models']['mahanghua'] = {
                'training_time': training_time,
                'n_samples': len(filtered_df),
                'n_features': X_transformed.shape[1],
                'n_classes': len(np.unique(y)),
                'feature_importance': None,
                'class_distribution': {str(i): int(count) for i, count in enumerate(np.bincount(y))},
                'condition': f"{self.condition_column}.startswith('{self.starts_with}')" if self.condition_column and self.starts_with else None
            }

            # Lưu feature importance nếu có
            if hasattr(classifier, 'feature_importances_'):
                try:
                    feature_names = self.feature_extractor.get_feature_names()
                    top_features = []

                    for i, importance in enumerate(classifier.feature_importances_):
                        if i < len(feature_names):
                            top_features.append({
                                'feature': feature_names[i],
                                'importance': float(importance)
                            })

                    # Sắp xếp theo độ quan trọng giảm dần và lấy top 20
                    top_features.sort(key=lambda x: x['importance'], reverse=True)
                    top_features = top_features[:20]

                    self.metadata['models']['mahanghua']['feature_importance'] = top_features
                except Exception as e:
                    logger.error(f"Lỗi khi lưu feature importance: {str(e)}")

        def train_outlier_detector(self, df: pd.DataFrame) -> None:
            """
            Huấn luyện mô hình phát hiện outlier

            Args:
                df: DataFrame chứa dữ liệu huấn luyện
            """
            logger.info(f"Bắt đầu huấn luyện mô hình phát hiện outlier cho khách hàng {self.customer_id}")

            # Kiểm tra cấu hình
            outlier_config = self.model_config.get('outlier_detection', {})
            enabled = outlier_config.get('enabled', True)

            if not enabled:
                logger.info("Phát hiện outlier bị tắt trong cấu hình")
                return

            start_time = time.time()

            # Tiền xử lý dữ liệu
            df_processed = self._preprocess_data(df)

            # Loại bỏ các cột mục tiêu
            X = df_processed.drop(columns=[self.primary_target, self.secondary_target], errors='ignore')

            # Tạo feature extractor nếu chưa có
            if self.feature_extractor is None:
                self.feature_extractor = self._create_feature_extractor()

            # Trích xuất đặc trưng
            logger.info("Trích xuất đặc trưng cho phát hiện outlier...")
            X_transformed = self.feature_extractor.fit_transform(X)
            logger.info(f"Đã trích xuất {X_transformed.shape[1]} đặc trưng")

            # Tạo mô hình phát hiện outlier
            outlier_detector = self._create_outlier_detector()
            if outlier_detector is None:
                logger.warning("Không thể tạo mô hình phát hiện outlier")
                return

            # Huấn luyện mô hình
            logger.info("Huấn luyện mô hình phát hiện outlier...")
            outlier_detector.fit(X_transformed)

            # Lưu mô hình
            self.outlier_model = outlier_detector

            # Thời gian huấn luyện
            training_time = time.time() - start_time
            logger.info(f"Hoàn tất huấn luyện mô hình phát hiện outlier trong {training_time:.2f} giây")

            # Thêm metadata
            self.metadata['models']['outlier'] = {
                'training_time': training_time,
                'n_samples': len(df),
                'n_features': X_transformed.shape[1],
                'method': outlier_config.get('method', 'isolation_forest'),
                'threshold': outlier_config.get('threshold', 0.85)
            }

        def save_models(self) -> Dict[str, str]:
            """
            Lưu các mô hình đã huấn luyện

            Returns:
                Dict chứa đường dẫn đến các file đã lưu
            """
            logger.info(f"Lưu các mô hình đã huấn luyện cho khách hàng {self.customer_id}")

            # Tạo thư mục lưu trữ
            model_dir = path_manager.get_customer_model_path(self.customer_id)

            saved_files = {}

            # Lưu mô hình HachToan
            if self.hachtoan_model is not None:
                hachtoan_dir = os.path.join(model_dir, 'hachtoan')
                os.makedirs(hachtoan_dir, exist_ok=True)

                # Lưu với phiên bản cụ thể
                hachtoan_path = os.path.join(hachtoan_dir, f'model_{self.version}.joblib')
                joblib.dump(self.hachtoan_model, hachtoan_path)

                # Lưu như phiên bản mới nhất
                latest_path = os.path.join(hachtoan_dir, 'model_latest.joblib')
                shutil.copy2(hachtoan_path, latest_path)

                logger.info(f"Đã lưu mô hình HachToan: {hachtoan_path}")
                saved_files['hachtoan_model'] = hachtoan_path

            # Lưu mô hình MaHangHoa
            if self.mahanghua_model is not None:
                mahanghua_dir = os.path.join(model_dir, 'mahanghua')
                os.makedirs(mahanghua_dir, exist_ok=True)

                # Lưu với phiên bản cụ thể
                mahanghua_path = os.path.join(mahanghua_dir, f'model_{self.version}.joblib')
                joblib.dump(self.mahanghua_model, mahanghua_path)

                # Lưu như phiên bản mới nhất
                latest_path = os.path.join(mahanghua_dir, 'model_latest.joblib')
                shutil.copy2(mahanghua_path, latest_path)

                logger.info(f"Đã lưu mô hình MaHangHoa: {mahanghua_path}")
                saved_files['mahanghua_model'] = mahanghua_path

            # Lưu mô hình phát hiện outlier
            if self.outlier_model is not None:
                outlier_dir = os.path.join(model_dir, 'outlier')
                os.makedirs(outlier_dir, exist_ok=True)

                # Lưu với phiên bản cụ thể
                outlier_path = os.path.join(outlier_dir, f'model_{self.version}.joblib')
                joblib.dump(self.outlier_model, outlier_path)

                # Lưu như phiên bản mới nhất
                latest_path = os.path.join(outlier_dir, 'model_latest.joblib')
                shutil.copy2(outlier_path, latest_path)

                logger.info(f"Đã lưu mô hình phát hiện outlier: {outlier_path}")
                saved_files['outlier_model'] = outlier_path

            # Lưu label encoders
            if self.label_encoders:
                encoders_dir = os.path.join(model_dir, 'encoders')
                os.makedirs(encoders_dir, exist_ok=True)

                # Lưu với phiên bản cụ thể
                encoders_path = os.path.join(encoders_dir, f'label_encoders_{self.version}.joblib')
                joblib.dump(self.label_encoders, encoders_path)

                # Lưu như phiên bản mới nhất
                latest_path = os.path.join(encoders_dir, 'label_encoders_latest.joblib')
                shutil.copy2(encoders_path, latest_path)

                logger.info(f"Đã lưu label encoders: {encoders_path}")
                saved_files['label_encoders'] = encoders_path

            # Lưu metadata
            metadata_dir = os.path.join(model_dir, 'metadata')
            os.makedirs(metadata_dir, exist_ok=True)

            # Cập nhật thời gian lưu vào metadata
            self.metadata['saved_timestamp'] = datetime.now().isoformat()
            self.metadata['saved_paths'] = saved_files

            # Lưu với phiên bản cụ thể
            metadata_path = os.path.join(metadata_dir, f'model_metadata_{self.version}.json')
            save_metadata(metadata_path, self.metadata)

            # Lưu như phiên bản mới nhất
            latest_path = os.path.join(metadata_dir, 'model_metadata_latest.json')
            with open(metadata_path, 'r', encoding='utf-8') as src, open(latest_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())

            logger.info(f"Đã lưu metadata: {metadata_path}")
            saved_files['metadata'] = metadata_path

            # Dọn dẹp các phiên bản cũ
            cleanup_old_model_versions(self.customer_id, 'hachtoan')
            cleanup_old_model_versions(self.customer_id, 'mahanghua')
            cleanup_old_model_versions(self.customer_id, 'outlier')

            return saved_files

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
                self.incremental_config = config_manager.get_customer_config(customer_id).get(
                    'incremental_training_config', {})

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
                logger.info(
                    f"Huấn luyện tăng cường XGBoost với {incremental_rounds} vòng lặp, learning_rate={incremental_lr}")

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
                logger.info(
                    f"Huấn luyện tăng cường LightGBM với {incremental_rounds} vòng lặp, learning_rate={incremental_lr}")

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
                                                                              getattr(base_model, 'n_estimators',
                                                                                      10) // 2)
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
                        logger.info(
                            f"Bắt đầu huấn luyện tăng cường mô hình MaHangHoa cho khách hàng {self.customer_id}")

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
                            logger.warning(
                                f"Không đủ dữ liệu để huấn luyện mô hình MaHangHoa: chỉ có {len(filtered_df)} dòng")
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
                                                                                      getattr(base_model,
                                                                                              'n_estimators', 10) // 2),
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

            def incremental_train_outlier_detector(self, df: pd.DataFrame):
                """
                Huấn luyện tăng cường mô hình phát hiện outlier

                Args:
                    df: DataFrame chứa dữ liệu huấn luyện mới
                """
                logger.info(
                    f"Bắt đầu huấn luyện tăng cường mô hình phát hiện outlier cho khách hàng {self.customer_id}")

                # Kiểm tra xem mô hình hiện có hỗ trợ huấn luyện tăng cường không
                # Phần lớn các mô hình phát hiện outlier không hỗ trợ huấn luyện tăng cường
                # nên thường sẽ huấn luyện lại mô hình

                # Huấn luyện mới
                trainer = ModelTrainer(self.customer_id)
                trainer.train_outlier_detector(df)
                self.outlier_model = trainer.outlier_model

                # Thêm metadata
                if self.outlier_model is not None:
                    self.metadata['models']['outlier'] = {
                        'incremental_training': False,  # Không phải huấn luyện tăng cường thực sự
                        'retrained': True,
                        'base_version': self.existing_metadata.get('version',
                                                                   'unknown') if self.existing_metadata else 'unknown',
                        'n_samples': len(df)
                    }

            def save_models(self) -> Dict[str, str]:
                """
                Lưu các mô hình đã huấn luyện tăng cường

                Returns:
                    Dict chứa đường dẫn đến các file đã lưu
                """
                logger.info(f"Lưu các mô hình đã huấn luyện tăng cường cho khách hàng {self.customer_id}")

                # Tạo thư mục lưu trữ
                model_dir = path_manager.get_customer_model_path(self.customer_id)

                saved_files = {}

                # Lưu mô hình HachToan
                if self.hachtoan_model is not None:
                    hachtoan_dir = os.path.join(model_dir, 'hachtoan')
                    os.makedirs(hachtoan_dir, exist_ok=True)

                    # Lưu với phiên bản cụ thể
                    hachtoan_path = os.path.join(hachtoan_dir, f'model_{self.version}.joblib')
                    joblib.dump(self.hachtoan_model, hachtoan_path)

                    # Lưu như phiên bản mới nhất
                    latest_path = os.path.join(hachtoan_dir, 'model_latest.joblib')
                    shutil.copy2(hachtoan_path, latest_path)

                    logger.info(f"Đã lưu mô hình HachToan: {hachtoan_path}")
                    saved_files['hachtoan_model'] = hachtoan_path
                elif self.existing_hachtoan_model is not None:
                    # Nếu không có mô hình mới, sử dụng mô hình hiện có
                    saved_files['hachtoan_model'] = os.path.join(model_dir, 'hachtoan', 'model_latest.joblib')

                # Lưu mô hình MaHangHoa
                if self.mahanghua_model is not None:
                    mahanghua_dir = os.path.join(model_dir, 'mahanghua')
                    os.makedirs(mahanghua_dir, exist_ok=True)

                    # Lưu với phiên bản cụ thể
                    mahanghua_path = os.path.join(mahanghua_dir, f'model_{self.version}.joblib')
                    joblib.dump(self.mahanghua_model, mahanghua_path)

                    # Lưu như phiên bản mới nhất
                    latest_path = os.path.join(mahanghua_dir, 'model_latest.joblib')
                    shutil.copy2(mahanghua_path, latest_path)

                    logger.info(f"Đã lưu mô hình MaHangHoa: {mahanghua_path}")
                    saved_files['mahanghua_model'] = mahanghua_path
                elif self.existing_mahanghua_model is not None:
                    # Nếu không có mô hình mới, sử dụng mô hình hiện có
                    saved_files['mahanghua_model'] = os.path.join(model_dir, 'mahanghua', 'model_latest.joblib')

                # Lưu mô hình phát hiện outlier
                if self.outlier_model is not None:
                    outlier_dir = os.path.join(model_dir, 'outlier')
                    os.makedirs(outlier_dir, exist_ok=True)

                    # Lưu với phiên bản cụ thể
                    outlier_path = os.path.join(outlier_dir, f'model_{self.version}.joblib')
                    joblib.dump(self.outlier_model, outlier_path)

                    # Lưu như phiên bản mới nhất
                    latest_path = os.path.join(outlier_dir, 'model_latest.joblib')
                    shutil.copy2(outlier_path, latest_path)

                    logger.info(f"Đã lưu mô hình phát hiện outlier: {outlier_path}")
                    saved_files['outlier_model'] = outlier_path
                elif self.existing_outlier_model is not None:
                    # Nếu không có mô hình mới, sử dụng mô hình hiện có
                    saved_files['outlier_model'] = os.path.join(model_dir, 'outlier', 'model_latest.joblib')

                # Lưu label encoders
                if self.label_encoders:
                    encoders_dir = os.path.join(model_dir, 'encoders')
                    os.makedirs(encoders_dir, exist_ok=True)

                    # Lưu với phiên bản cụ thể
                    encoders_path = os.path.join(encoders_dir, f'label_encoders_{self.version}.joblib')
                    joblib.dump(self.label_encoders, encoders_path)

                    # Lưu như phiên bản mới nhất
                    latest_path = os.path.join(encoders_dir, 'label_encoders_latest.joblib')
                    shutil.copy2(encoders_path, latest_path)

                    logger.info(f"Đã lưu label encoders: {encoders_path}")
                    saved_files['label_encoders'] = encoders_path
                elif self.existing_label_encoders is not None:
                    # Nếu không có encoder mới, sử dụng encoder hiện có
                    saved_files['label_encoders'] = os.path.join(model_dir, 'encoders', 'label_encoders_latest.joblib')

                # Lưu metadata
                metadata_dir = os.path.join(model_dir, 'metadata')
                os.makedirs(metadata_dir, exist_ok=True)

                # Cập nhật thời gian lưu vào metadata
                self.metadata['saved_timestamp'] = datetime.now().isoformat()
                self.metadata['saved_paths'] = saved_files

                # Lưu với phiên bản cụ thể
                metadata_path = os.path.join(metadata_dir, f'model_metadata_{self.version}.json')
                save_metadata(metadata_path, self.metadata)

                # Lưu như phiên bản mới nhất
                latest_path = os.path.join(metadata_dir, 'model_metadata_latest.json')
                with open(metadata_path, 'r', encoding='utf-8') as src, open(latest_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())

                logger.info(f"Đã lưu metadata: {metadata_path}")
                saved_files['metadata'] = metadata_path

                # Dọn dẹp các phiên bản cũ
                cleanup_old_model_versions(self.customer_id, 'hachtoan')
                cleanup_old_model_versions(self.customer_id, 'mahanghua')
                cleanup_old_model_versions(self.customer_id, 'outlier')

                return saved_files

            def train_customer_model(customer_id: str, train_file: str, test_file: str = None,
                                     incremental: bool = False, version: str = None) -> Dict[str, Any]:
                """
                Huấn luyện mô hình cho khách hàng

                Args:
                    customer_id: ID của khách hàng
                    train_file: Đường dẫn đến file CSV dữ liệu huấn luyện
                    test_file: Đường dẫn đến file CSV dữ liệu kiểm tra (optional)
                    incremental: Nếu True, sẽ thực hiện huấn luyện tăng cường
                    version: Phiên bản mô hình (nếu None, sẽ tạo mới)

                Returns:
                    Dict chứa thông tin về kết quả huấn luyện
                """
                start_time = time.time()
                logger.info(f"Bắt đầu huấn luyện mô hình cho khách hàng {customer_id}")

                try:
                    # Đọc dữ liệu huấn luyện
                    train_df = pd.read_csv(train_file, sep=";", encoding='utf-8-sig')
                    logger.info(f"Đã đọc {len(train_df)} dòng dữ liệu huấn luyện từ {train_file}")

                    # Đọc dữ liệu kiểm tra nếu có
                    test_df = None
                    if test_file:
                        try:
                            test_df = pd.read_csv(test_file, sep=";", encoding='utf-8-sig')
                            logger.info(f"Đã đọc {len(test_df)} dòng dữ liệu kiểm tra từ {test_file}")
                        except Exception as e:
                            logger.error(f"Lỗi khi đọc file kiểm tra {test_file}: {str(e)}")

                    # Huấn luyện mô hình dựa trên chế độ
                    if incremental:
                        # Huấn luyện tăng cường
                        trainer = IncrementalModelTrainer(customer_id, version)

                        # Huấn luyện các mô hình
                        trainer.incremental_train_hachtoan_model(train_df)
                        trainer.incremental_train_mahanghua_model(train_df)
                        trainer.incremental_train_outlier_detector(train_df)
                    else:
                        # Huấn luyện mới
                        trainer = ModelTrainer(customer_id, version)

                        # Huấn luyện các mô hình
                        trainer.train_hachtoan_model(train_df)
                        trainer.train_mahanghua_model(train_df)
                        trainer.train_outlier_detector(train_df)

                    # Lưu các mô hình
                    saved_files = trainer.save_models()

                    # Thời gian huấn luyện
                    total_time = time.time() - start_time
                    logger.info(f"Hoàn tất huấn luyện trong {total_time:.2f} giây")

                    # Tạo kết quả
                    result = {
                        "status": "success",
                        "customer_id": customer_id,
                        "train_file": train_file,
                        "test_file": test_file,
                        "incremental": incremental,
                        "version": trainer.version,
                        "saved_files": saved_files,
                        "total_time": total_time,
                        "timestamp": datetime.now().isoformat()
                    }

                    return result

                except Exception as e:
                    logger.exception(f"Lỗi khi huấn luyện mô hình: {str(e)}")
                    return {
                        "status": "error",
                        "customer_id": customer_id,
                        "train_file": train_file,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }

            def main(self):
                """Hàm chính để chạy script từ command line"""
                parser = argparse.ArgumentParser(description='Huấn luyện mô hình phân loại')
                parser.add_argument('--customer-id', required=True, help='ID của khách hàng')
                parser.add_argument('--train-file', required=True, help='Đường dẫn đến file CSV dữ liệu huấn luyện')
                parser.add_argument('--test-file', help='Đường dẫn đến file CSV dữ liệu kiểm tra (tùy chọn)')
                parser.add_argument('--incremental', action='store_true', help='Thực hiện huấn luyện tăng cường')
                parser.add_argument('--version', help='Phiên bản mô hình (tùy chọn)')
                parser.add_argument('--output-file', help='Đường dẫn đến file JSON kết quả (tùy chọn)')

                args = parser.parse_args()

                try:
                    result = self.train_customer_model(
                        customer_id=args.customer_id,
                        train_file=args.train_file,
                        test_file=args.test_file,
                        incremental=args.incremental,
                        version=args.version
                    )

                    # Lưu kết quả vào file JSON nếu được chỉ định
                    if args.output_file:
                        with open(args.output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        logger.info(f"Đã lưu kết quả vào: {args.output_file}")

                    # In ra kết quả cho script gọi
                    print(json.dumps(result))

                    # Trả về mã lỗi
                    return 0 if result["status"] == "success" else 1
                except Exception as e:
                    logger.exception(f"Lỗi không xử lý được: {str(e)}")
                    return 1

            if __name__ == "__main__":
                sys.exit(main())

