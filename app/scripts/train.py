#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script huấn luyện mô hình
- Huấn luyện mô hình dự đoán HachToan
- Huấn luyện mô hình dự đoán MaHangHoa (nếu cần)
- Huấn luyện mô hình phát hiện outlier
- Lưu mô hình và metadata
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import các thư viện cần thiết
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Import modules của dự án
from app.config import config_manager, path_manager, constants
from app.config.utils import save_metadata, load_metadata, generate_model_version, cleanup_old_model_versions

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
    Lớp huấn luyện mô hình cho bài toán phân loại đa lớp
    - Hỗ trợ mô hình dự đoán HachToan và MaHangHoa
    - Hỗ trợ xử lý dữ liệu mất cân bằng
    - Hỗ trợ phát hiện outlier
    """

    def __init__(self, customer_id: str):
        """
        Khởi tạo ModelTrainer

        Args:
            customer_id: ID của khách hàng
        """
        self.customer_id = customer_id

        # Tải cấu hình
        self.column_config = config_manager.get_column_config(customer_id)
        self.preprocess_config = config_manager.get_preprocessing_config(customer_id)
        self.model_config = config_manager.get_customer_config(customer_id).get('model_config', {})
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

        # Các encoder và transformer
        self.text_vectorizer = None
        self.id_encoder = None
        self.label_encoders = {}
        self.feature_names = None

        # Các mô hình
        self.hachtoan_model = None
        self.mahanghua_model = None
        self.outlier_model = None

        # Metadata
        self.metadata = {
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat(),
            "version": generate_model_version(),
            "models": {}
        }

    def _create_text_vectorizer(self):
        """Tạo text vectorizer dựa trên cấu hình"""
        text_config = self.preprocess_config.get('text_features', {})
        max_features = text_config.get('max_features', 10000)
        ngram_range = tuple(text_config.get('ngram_range', [1, 2]))
        min_df = text_config.get('min_df', 2)

        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            strip_accents='unicode'
        )

    def _create_id_encoder(self):
        """Tạo encoder cho các cột ID"""
        return OneHotEncoder(
            handle_unknown=self.preprocess_config.get('id_features', {}).get('handle_unknown', 'ignore'),
            sparse=False
        )

    def _create_feature_transformer(self):
        """Tạo transformer cho các đặc trưng đầu vào"""
        transformers = []

        # Xử lý cột văn bản tiếng Việt
        if self.vietnamese_text_columns:
            self.text_vectorizer = self._create_text_vectorizer()
            for col in self.vietnamese_text_columns:
                transformers.append((f'text_{col}', self.text_vectorizer, col))

        # Xử lý cột ID
        if self.id_columns:
            self.id_encoder = self._create_id_encoder()
            for col in self.id_columns:
                transformers.append((f'id_{col}', self.id_encoder, col))

        # Tạo và trả về transformer
        return ColumnTransformer(transformers)

    def _create_imbalance_handler(self, model_type: str):
        """
        Tạo bộ xử lý dữ liệu mất cân bằng dựa trên cấu hình

        Args:
            model_type: Loại mô hình ('hachtoan_model' hoặc 'mahanghua_model')

        Returns:
            Bộ xử lý dữ liệu mất cân bằng hoặc None nếu không cần
        """
        # Kiểm tra xem có cần xử lý mất cân bằng không
        handle_imbalance = self.model_config.get(model_type, {}).get('handle_imbalance', True)

        if not handle_imbalance:
            return None

        # Lấy chiến lược xử lý mất cân bằng
        strategy = self.model_config.get(model_type, {}).get('imbalance_strategy', 'auto')

        if strategy == 'smote':
            return SMOTE(random_state=42)
        elif strategy == 'adasyn':
            return ADASYN(random_state=42)
        elif strategy == 'smoteenn':
            return SMOTEENN(random_state=42)
        elif strategy == 'smotetomek':
            return SMOTETomek(random_state=42)
        elif strategy == 'undersampling':
            return RandomUnderSampler(random_state=42)
        elif strategy == 'auto':
            # Mặc định sử dụng SMOTEENN cho mất cân bằng nghiêm trọng
            return SMOTEENN(random_state=42)

        return None

    def _create_model(self, model_type: str):
        """
        Tạo mô hình dựa trên cấu hình

        Args:
            model_type: Loại mô hình ('hachtoan_model' hoặc 'mahanghua_model')

        Returns:
            Mô hình đã được cấu hình
        """
        model_type_config = self.model_config.get(model_type, {})
        model_algorithm = model_type_config.get('type', 'xgboost')
        params = model_type_config.get('params', {})

        if model_algorithm == 'xgboost':
            # Mặc định cho multi-class classification
            default_params = {
                'objective': 'multi:softprob',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }

            # Cập nhật từ cấu hình
            for key, value in params.items():
                default_params[key] = value

            return xgb.XGBClassifier(**default_params)

        elif model_algorithm == 'lightgbm':
            default_params = {
                'objective': 'multiclass',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }

            for key, value in params.items():
                default_params[key] = value

            return lgb.LGBMClassifier(**default_params)

        elif model_algorithm == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }

            for key, value in params.items():
                default_params[key] = value

            return RandomForestClassifier(**default_params)

        else:
            # Mặc định nếu không nhận dạng được thuật toán
            logger.warning(f"Không nhận dạng được thuật toán {model_algorithm}, sử dụng XGBoost mặc định")
            return xgb.XGBClassifier(
                objective='multi:softprob',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

    def _create_outlier_detector(self):
        """
        Tạo mô hình phát hiện outlier dựa trên cấu hình

        Returns:
            Mô hình phát hiện outlier
        """
        outlier_config = self.model_config.get('outlier_detection', {})
        outlier_method = outlier_config.get('method', 'isolation_forest')
        params = outlier_config.get('params', {})

        if outlier_method == 'isolation_forest':
            default_params = {
                'n_estimators': 100,
                'contamination': 'auto',
                'random_state': 42
            }

            for key, value in params.items():
                default_params[key] = value

            return IsolationForest(**default_params)

        # Có thể thêm các phương pháp khác ở đây

        else:
            logger.warning(
                f"Không nhận dạng được phương pháp phát hiện outlier {outlier_method}, sử dụng Isolation Forest mặc định")
            return IsolationForest(n_estimators=100, contamination='auto', random_state=42)

    def _build_pipeline(self, model_type: str, feature_transformer):
        """
        Xây dựng pipeline huấn luyện hoàn chỉnh

        Args:
            model_type: Loại mô hình ('hachtoan_model' hoặc 'mahanghua_model')
            feature_transformer: Transformer xử lý đặc trưng

        Returns:
            Pipeline huấn luyện
        """
        # Tạo mô hình
        model = self._create_model(model_type)

        # Tạo bộ xử lý dữ liệu mất cân bằng
        imbalance_handler = self._create_imbalance_handler(model_type)

        # Xây dựng pipeline
        steps = [
            ('features', feature_transformer),
            ('classifier', model)
        ]

        # Thêm bộ xử lý dữ liệu mất cân bằng nếu cần
        if imbalance_handler:
            steps.insert(1, ('imbalance', imbalance_handler))
            return ImbPipeline(steps)

        return Pipeline(steps)

    def train_hachtoan_model(self, train_df: pd.DataFrame):
        """
        Huấn luyện mô hình dự đoán HachToan

        Args:
            train_df: DataFrame chứa dữ liệu huấn luyện
        """
        logger.info(f"Bắt đầu huấn luyện mô hình HachToan cho khách hàng {self.customer_id}")

        # Chuẩn bị đặc trưng và nhãn
        X = train_df.drop(columns=[self.primary_target, self.secondary_target], errors='ignore')
        y = train_df[self.primary_target]

        # Mã hóa nhãn nếu cần
        if not y.dtype.name == 'int64' and not y.dtype.name == 'int32':
            logger.info(f"Mã hóa nhãn HachToan từ kiểu {y.dtype}")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self.label_encoders['hachtoan'] = label_encoder

        # Tạo transformer đặc trưng
        feature_transformer = self._create_feature_transformer()

        # Xây dựng pipeline
        pipeline = self._build_pipeline('hachtoan_model', feature_transformer)

        # Cross-validation nếu được cấu hình
        if self.training_config.get('cross_validation', {}).get('enabled', True):
            n_splits = self.training_config.get('cross_validation', {}).get('n_splits', 5)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            logger.info(f"Thực hiện cross-validation với {n_splits} folds")
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')

            logger.info(f"Kết quả cross-validation: {cv_scores}")
            logger.info(f"F1 trung bình: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            self.metadata['models']['hachtoan'] = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std())
            }

        # Huấn luyện trên toàn bộ dữ liệu
        logger.info("Huấn luyện mô hình trên toàn bộ dữ liệu")
        start_time = time.time()
        pipeline.fit(X, y)
        training_time = time.time() - start_time

        logger.info(f"Đã huấn luyện xong mô hình HachToan trong {training_time:.2f} giây")

        # Trích xuất thông tin quan trọng nếu có
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            self._extract_feature_importance(pipeline, 'hachtoan_model')

        # Lưu mô hình
        self.hachtoan_model = pipeline

        # Thêm metadata
        self.metadata['models']['hachtoan'].update({
            'training_time': training_time,
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'n_classes': len(np.unique(y)),
            'algorithm': self.model_config.get('hachtoan_model', {}).get('type', 'xgboost')
        })

    def train_mahanghua_model(self, train_df: pd.DataFrame):
        """
        Huấn luyện mô hình dự đoán MaHangHoa

        Args:
            train_df: DataFrame chứa dữ liệu huấn luyện
        """
        logger.info(f"Bắt đầu huấn luyện mô hình MaHangHoa cho khách hàng {self.customer_id}")

        # Lọc dữ liệu theo điều kiện
        if self.condition_column and self.starts_with:
            condition_mask = train_df[self.condition_column].astype(str).str.startswith(self.starts_with)
            if condition_mask.sum() == 0:
                logger.warning(
                    f"Không có dữ liệu nào thỏa điều kiện {self.condition_column}.startswith('{self.starts_with}')")
                logger.warning("Bỏ qua huấn luyện mô hình MaHangHoa")
                return

            logger.info(f"Lọc dữ liệu theo điều kiện: {condition_mask.sum()}/{len(train_df)} dòng thỏa mãn")
            filtered_df = train_df[condition_mask].reset_index(drop=True)
        else:
            # Nếu không có điều kiện, sử dụng toàn bộ dữ liệu
            filtered_df = train_df

        # Kiểm tra xem có đủ dữ liệu để huấn luyện không
        if len(filtered_df) < 10:
            logger.warning(f"Không đủ dữ liệu để huấn luyện mô hình MaHangHoa: chỉ có {len(filtered_df)} dòng")
            return

        # Chuẩn bị đặc trưng và nhãn
        X = filtered_df.drop(columns=[self.secondary_target], errors='ignore')
        y = filtered_df[self.secondary_target]

        # Mã hóa nhãn nếu cần
        if not y.dtype.name == 'int64' and not y.dtype.name == 'int32':
            logger.info(f"Mã hóa nhãn MaHangHoa từ kiểu {y.dtype}")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self.label_encoders['mahanghua'] = label_encoder

        # Tạo transformer đặc trưng
        feature_transformer = self._create_feature_transformer()

        # Xây dựng pipeline
        pipeline = self._build_pipeline('mahanghua_model', feature_transformer)

        # Cross-validation nếu được cấu hình
        if self.training_config.get('cross_validation', {}).get('enabled', True):
            n_splits = min(self.training_config.get('cross_validation', {}).get('n_splits', 5),
                           len(filtered_df) // 10)  # Đảm bảo đủ dữ liệu cho mỗi fold

            if n_splits >= 2:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

                logger.info(f"Thực hiện cross-validation với {n_splits} folds")
                cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')

                logger.info(f"Kết quả cross-validation: {cv_scores}")
                logger.info(f"F1 trung bình: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

                self.metadata['models']['mahanghua'] = {
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std())
                }

        # Huấn luyện trên toàn bộ dữ liệu
        logger.info("Huấn luyện mô hình trên toàn bộ dữ liệu")
        start_time = time.time()
        pipeline.fit(X, y)
        training_time = time.time() - start_time

        logger.info(f"Đã huấn luyện xong mô hình MaHangHoa trong {training_time:.2f} giây")

        # Trích xuất thông tin quan trọng nếu có
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            self._extract_feature_importance(pipeline, 'mahanghua_model')

        # Lưu mô hình
        self.mahanghua_model = pipeline

        # Thêm metadata
        if 'mahanghua' not in self.metadata['models']:
            self.metadata['models']['mahanghua'] = {}

        self.metadata['models']['mahanghua'].update({
            'training_time': training_time,
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'n_classes': len(np.unique(y)),
            'algorithm': self.model_config.get('mahanghua_model', {}).get('type', 'xgboost'),
            'condition': f"{self.condition_column}.startswith('{self.starts_with}')" if self.condition_column and self.starts_with else None
        })

    def train_outlier_detector(self, train_df: pd.DataFrame):
        """
        Huấn luyện mô hình phát hiện outlier

        Args:
            train_df: DataFrame chứa dữ liệu huấn luyện
        """
        # Kiểm tra cấu hình phát hiện outlier
        outlier_config = self.model_config.get('outlier_detection', {})
        if not outlier_config.get('enabled', True):
            logger.info("Phát hiện outlier bị tắt trong cấu hình, bỏ qua")
            return

        logger.info(f"Bắt đầu huấn luyện mô hình phát hiện outlier cho khách hàng {self.customer_id}")

        # Chuẩn bị đặc trưng
        X = train_df.drop(columns=[self.primary_target, self.secondary_target], errors='ignore')

        # Tạo transformer đặc trưng
        feature_transformer = self._create_feature_transformer()

        # Tạo mô hình phát hiện outlier
        outlier_detector = self._create_outlier_detector()

        # Xây dựng pipeline
        pipeline = Pipeline([
            ('features', feature_transformer),
            ('outlier_detector', outlier_detector)
        ])

        # Huấn luyện mô hình
        logger.info("Huấn luyện mô hình phát hiện outlier")
        start_time = time.time()
        pipeline.fit(X)
        training_time = time.time() - start_time

        logger.info(f"Đã huấn luyện xong mô hình phát hiện outlier trong {training_time:.2f} giây")

        # Lưu mô hình
        self.outlier_model = pipeline

        # Thêm metadata
        self.metadata['models']['outlier'] = {
            'training_time': training_time,
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'method': outlier_config.get('method', 'isolation_forest'),
            'threshold': outlier_config.get('threshold', 0.85)
        }

    def _extract_feature_importance(self, pipeline, model_type: str):
        """
        Trích xuất và lưu thông tin feature importance từ mô hình

        Args:
            pipeline: Pipeline chứa mô hình
            model_type: Loại mô hình ('hachtoan_model' hoặc 'mahanghua_model')
        """
        logger.info(f"Trích xuất feature importance cho mô hình {model_type}")

        try:
            # Lấy mô hình từ pipeline
            model = pipeline.named_steps['classifier']

            # Lấy feature importance
            feature_importances = model.feature_importances_

            # Lấy tên các đặc trưng
            feature_names = []
            for name, transformer, column in pipeline.named_steps['features'].transformers_:
                # TfidfVectorizer và CountVectorizer có thuộc tính get_feature_names_out
                if hasattr(transformer, 'get_feature_names_out'):
                    # Thêm tên cột vào tên đặc trưng
                    names = [f"{column}_{feat}" for feat in transformer.get_feature_names_out()]
                    feature_names.extend(names)
                # OneHotEncoder có thuộc tính categories_
                elif hasattr(transformer, 'categories_'):
                    for cat in transformer.categories_:
                        names = [f"{column}_{cat[i]}" for i in range(len(cat))]
                        feature_names.extend(names)
                else:
                    # Nếu không có cách nào để lấy tên đặc trưng
                    feature_names.append(f"{name}")

            # Tạo danh sách các đặc trưng quan trọng
            if len(feature_names) == len(feature_importances):
                # Sắp xếp theo độ quan trọng giảm dần
                feature_importance = sorted(zip(feature_names, feature_importances),
                                            key=lambda x: x[1], reverse=True)

                # Lưu vào metadata
                model_key = 'hachtoan' if model_type == 'hachtoan_model' else 'mahanghua'
                self.metadata['models'][model_key]['feature_importance'] = [
                    {'feature': feat, 'importance': float(imp)}
                    for feat, imp in feature_importance[:20]  # Chỉ lưu 20 đặc trưng quan trọng nhất
                ]

                logger.info(f"Top 5 đặc trưng quan trọng nhất cho {model_type}:")
                for i, (feat, imp) in enumerate(feature_importance[:5]):
                    logger.info(f"  {i + 1}. {feat}: {imp:.4f}")
            else:
                logger.warning(f"Không thể trích xuất feature importance: số lượng đặc trưng không khớp")

        except Exception as e:
            logger.warning(f"Lỗi khi trích xuất feature importance: {str(e)}")

    def evaluate_models(self, test_df: pd.DataFrame):
        """
        Đánh giá hiệu suất của các mô hình trên tập kiểm tra

        Args:
            test_df: DataFrame chứa dữ liệu kiểm tra
        """
        logger.info(f"Đánh giá hiệu suất mô hình cho khách hàng {self.customer_id}")

        # Đánh giá mô hình HachToan
        if self.hachtoan_model is not None:
            self._evaluate_model(test_df, 'hachtoan')

        # Đánh giá mô hình MaHangHoa
        if self.mahanghua_model is not None:
            # Lọc dữ liệu theo điều kiện nếu cần
            if self.condition_column and self.starts_with:
                condition_mask = test_df[self.condition_column].astype(str).str.startswith(self.starts_with)
                if condition_mask.sum() > 0:
                    filtered_df = test_df[condition_mask].reset_index(drop=True)
                    self._evaluate_model(filtered_df, 'mahanghua')
                else:
                    logger.warning("Không có dữ liệu kiểm tra thỏa điều kiện cho MaHangHoa")
            else:
                self._evaluate_model(test_df, 'mahanghua')

    def _evaluate_model(self, test_df: pd.DataFrame, model_type: str):
        """
        Đánh giá hiệu suất của một mô hình cụ thể

        Args:
            test_df: DataFrame chứa dữ liệu kiểm tra
            model_type: Loại mô hình ('hachtoan' hoặc 'mahanghua')
        """
        logger.info(f"Đánh giá mô hình {model_type}")

        # Lấy mô hình phù hợp
        if model_type == 'hachtoan':
            model = self.hachtoan_model
            target_column = self.primary_target
        else:  # mahanghua
            model = self.mahanghua_model
            target_column = self.secondary_target

        if model is None:
            logger.warning(f"Không có mô hình {model_type} để đánh giá")
            return

        # Chuẩn bị dữ liệu
        X = test_df.drop(columns=[self.primary_target, self.secondary_target], errors='ignore')
        y_true = test_df[target_column]

        # Mã hóa nhãn nếu cần
        if model_type in self.label_encoders:
            y_true = self.label_encoders[model_type].transform(y_true)

        # Dự đoán
        y_pred = model.predict(X)

        # Lấy xác suất nếu có
        try:
            y_prob = model.predict_proba(X)
            has_probabilities = True
        except:
            has_probabilities = False

        # Tính các chỉ số đánh giá
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info(f"Precision (weighted): {precision:.4f}")
        logger.info(f"Recall (weighted): {recall:.4f}")

        # Tạo báo cáo phân loại
        class_report = classification_report(y_true, y_pred, output_dict=True)
        logger.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")

        # Lưu kết quả đánh giá vào metadata
        evaluation = {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'class_report': class_report,
            'n_test_samples': len(test_df)
        }

        # Thêm AUC nếu có xác suất
        if has_probabilities and len(np.unique(y_true)) == 2:  # Chỉ tính AUC cho binary classification
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_prob[:, 1])
            evaluation['auc'] = float(auc)
            logger.info(f"AUC: {auc:.4f}")

        self.metadata['models'][model_type]['evaluation'] = evaluation

    def save_models(self, output_dir: str = None):
        """
        Lưu các mô hình đã huấn luyện

        Args:
            output_dir: Thư mục đầu ra (nếu None, sẽ sử dụng thư mục mặc định)

        Returns:
            Dict chứa đường dẫn đến các file mô hình đã lưu
        """
        logger.info(f"Lưu các mô hình cho khách hàng {self.customer_id}")

        # Đảm bảo đã có version
        version = self.metadata.get('version', generate_model_version())

        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = path_manager.get_customer_model_path(self.customer_id)

        os.makedirs(output_dir, exist_ok=True)

        saved_models = {}

        # Lưu mô hình HachToan
        if self.hachtoan_model is not None:
            hachtoan_dir = os.path.join(output_dir, 'hachtoan')
            os.makedirs(hachtoan_dir, exist_ok=True)

            hachtoan_path = os.path.join(hachtoan_dir, f'model_{version}.joblib')
            joblib.dump(self.hachtoan_model, hachtoan_path)
            logger.info(f"Đã lưu mô hình HachToan: {hachtoan_path}")

            # Lưu phiên bản mới nhất
            latest_path = os.path.join(hachtoan_dir, 'model_latest.joblib')
            joblib.dump(self.hachtoan_model, latest_path)

            saved_models['hachtoan'] = hachtoan_path

            # Xóa các phiên bản cũ nếu cần
            cleanup_old_model_versions(self.customer_id, 'hachtoan')

        # Lưu mô hình MaHangHoa
        if self.mahanghua_model is not None:
            mahanghua_dir = os.path.join(output_dir, 'mahanghua')
            os.makedirs(mahanghua_dir, exist_ok=True)

            mahanghua_path = os.path.join(mahanghua_dir, f'model_{version}.joblib')
            joblib.dump(self.mahanghua_model, mahanghua_path)
            logger.info(f"Đã lưu mô hình MaHangHoa: {mahanghua_path}")

            # Lưu phiên bản mới nhất
            latest_path = os.path.join(mahanghua_dir, 'model_latest.joblib')
            joblib.dump(self.mahanghua_model, latest_path)

            saved_models['mahanghua'] = mahanghua_path

            # Xóa các phiên bản cũ nếu cần
            cleanup_old_model_versions(self.customer_id, 'mahanghua')

        # Lưu mô hình phát hiện outlier
        if self.outlier_model is not None:
            outlier_dir = os.path.join(output_dir, 'outlier')
            os.makedirs(outlier_dir, exist_ok=True)

            outlier_path = os.path.join(outlier_dir, f'model_{version}.joblib')
            joblib.dump(self.outlier_model, outlier_path)
            logger.info(f"Đã lưu mô hình phát hiện outlier: {outlier_path}")

            # Lưu phiên bản mới nhất
            latest_path = os.path.join(outlier_dir, 'model_latest.joblib')
            joblib.dump(self.outlier_model, latest_path)

            saved_models['outlier'] = outlier_path

            # Xóa các phiên bản cũ nếu cần
            cleanup_old_model_versions(self.customer_id, 'outlier')

        # Lưu các bộ encoder
        if self.label_encoders:
            encoders_dir = os.path.join(output_dir, 'encoders')
            os.makedirs(encoders_dir, exist_ok=True)

            encoders_path = os.path.join(encoders_dir, f'label_encoders_{version}.joblib')
            joblib.dump(self.label_encoders, encoders_path)
            logger.info(f"Đã lưu các label encoders: {encoders_path}")

            # Lưu phiên bản mới nhất
            latest_path = os.path.join(encoders_dir, 'label_encoders_latest.joblib')
            joblib.dump(self.label_encoders, latest_path)

            saved_models['encoders'] = encoders_path

        # Lưu metadata
        metadata_dir = os.path.join(output_dir, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        metadata_path = os.path.join(metadata_dir, f'model_metadata_{version}.json')
        save_metadata(metadata_path, self.metadata)
        logger.info(f"Đã lưu metadata: {metadata_path}")

        # Lưu phiên bản mới nhất
        latest_path = os.path.join(metadata_dir, 'model_metadata_latest.json')
        save_metadata(latest_path, self.metadata)

        saved_models['metadata'] = metadata_path

        return saved_models

    def train_and_save(self, train_file: str, test_file: str = None, output_dir: str = None):
        """
        Huấn luyện và lưu toàn bộ mô hình

        Args:
            train_file: Đường dẫn đến file dữ liệu huấn luyện
            test_file: Đường dẫn đến file dữ liệu kiểm tra (nếu có)
            output_dir: Thư mục đầu ra (nếu None, sẽ sử dụng thư mục mặc định)

        Returns:
            Dict chứa đường dẫn đến các file mô hình đã lưu
        """
        logger.info(f"Bắt đầu huấn luyện mô hình cho khách hàng {self.customer_id}")
        logger.info(f"Tập huấn luyện: {train_file}")

        # Đọc dữ liệu huấn luyện
        try:
            train_df = pd.read_csv(train_file, encoding='utf-8-sig')
            logger.info(f"Đã đọc tập huấn luyện: {len(train_df)} dòng")
        except Exception as e:
            logger.error(f"Lỗi khi đọc tập huấn luyện: {str(e)}")
            try:
                train_df = pd.read_csv(train_file, encoding='latin1')
                logger.info(f"Đã đọc tập huấn luyện với encoding latin1: {len(train_df)} dòng")
            except Exception as e2:
                logger.error(f"Không thể đọc tập huấn luyện: {str(e2)}")
                raise ValueError(f"Không thể đọc tập huấn luyện: {str(e2)}")

        # Đọc dữ liệu kiểm tra nếu có
        test_df = None
        if test_file:
            try:
                test_df = pd.read_csv(test_file, encoding='utf-8-sig')
                logger.info(f"Đã đọc tập kiểm tra: {len(test_df)} dòng")
            except Exception as e:
                logger.error(f"Lỗi khi đọc tập kiểm tra: {str(e)}")
                try:
                    test_df = pd.read_csv(test_file, encoding='latin1')
                    logger.info(f"Đã đọc tập kiểm tra với encoding latin1: {len(test_df)} dòng")
                except Exception as e2:
                    logger.error(f"Không thể đọc tập kiểm tra: {str(e2)}")
                    logger.warning("Tiếp tục mà không có tập kiểm tra")

        # Huấn luyện các mô hình
        try:
            # 1. Huấn luyện mô hình HachToan
            self.train_hachtoan_model(train_df)

            # 2. Huấn luyện mô hình MaHangHoa nếu có cột mục tiêu
            if self.secondary_target and self.secondary_target in train_df.columns:
                self.train_mahanghua_model(train_df)

            # 3. Huấn luyện mô hình phát hiện outlier
            self.train_outlier_detector(train_df)

            # 4. Đánh giá mô hình nếu có tập kiểm tra
            if test_df is not None:
                self.evaluate_models(test_df)

            # 5. Lưu các mô hình
            return self.save_models(output_dir)

        except Exception as e:
            logger.exception(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            raise

    # Các hàm helper ở cấp module
    def train_customer_model(customer_id: str, train_file: str, test_file: str = None, output_dir: str = None):
        """
        Huấn luyện mô hình cho một khách hàng cụ thể

        Args:
            customer_id: ID của khách hàng
            train_file: Đường dẫn đến file dữ liệu huấn luyện
            test_file: Đường dẫn đến file dữ liệu kiểm tra (nếu có)
            output_dir: Thư mục đầu ra (nếu None, sẽ sử dụng thư mục mặc định)

        Returns:
            Dict chứa đường dẫn đến các file mô hình đã lưu
        """
        trainer = ModelTrainer(customer_id)
        return trainer.train_and_save(train_file, test_file, output_dir)

    def incremental_train_customer_model(customer_id: str, new_data_file: str, output_dir: str = None):
        """
        Huấn luyện tăng cường mô hình cho một khách hàng cụ thể

        Args:
            customer_id: ID của khách hàng
            new_data_file: Đường dẫn đến file dữ liệu mới
            output_dir: Thư mục đầu ra (nếu None, sẽ sử dụng thư mục mặc định)

        Returns:
            Dict chứa đường dẫn đến các file mô hình đã lưu
        """
        # TODO: Implement incremental training
        # Hiện tại, thực hiện huấn luyện lại trên toàn bộ dữ liệu
        # Trong tương lai, cần cải thiện để huấn luyện tăng cường thật sự

        logger.warning("Huấn luyện tăng cường chưa được hỗ trợ, thực hiện huấn luyện lại trên dữ liệu mới")
        trainer = ModelTrainer(customer_id)
        return trainer.train_and_save(new_data_file, output_dir=output_dir)

    # Hàm main
    def main(self):
        """Hàm chính để chạy script từ command line"""
        parser = argparse.ArgumentParser(description='Huấn luyện mô hình phân loại')
        parser.add_argument('--customer-id', required=True, help='ID của khách hàng')
        parser.add_argument('--train-file', required=True, help='Đường dẫn đến file dữ liệu huấn luyện')
        parser.add_argument('--test-file', help='Đường dẫn đến file dữ liệu kiểm tra (tùy chọn)')
        parser.add_argument('--output-dir', help='Thư mục đầu ra (tùy chọn)')
        parser.add_argument('--incremental', action='store_true', help='Huấn luyện tăng cường')

        args = parser.parse_args()

        try:
            if args.incremental:
                result = self.incremental_train_customer_model(
                    customer_id=args.customer_id,
                    new_data_file=args.train_file,
                    output_dir=args.output_dir
                )
            else:
                result = self.train_customer_model(
                    customer_id=args.customer_id,
                    train_file=args.train_file,
                    test_file=args.test_file,
                    output_dir=args.output_dir
                )

            logger.info("Huấn luyện mô hình hoàn tất")

            # In ra thông tin để script gọi có thể sử dụng
            print(json.dumps(result))

            return 0
        except Exception as e:
            logger.exception(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            return 1

    # Phần thực thi khi chạy trực tiếp
    if __name__ == "__main__":
        sys.exit(main())