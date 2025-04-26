#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script dự đoán sử dụng mô hình đã huấn luyện
- Dự đoán HachToan từ MSTNguoiBan và TenHangHoaDichVu
- Nếu HachToan bắt đầu bằng "15" thì dự đoán MaHangHoa
- Phát hiện outlier và đưa ra cảnh báo
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

from app.config import config_manager, path_manager, constants
from app.config.utils import load_metadata, save_metadata, normalize_vietnamese_text

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='predict.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('predict')


class ModelPredictor:
    """
    Lớp dự đoán sử dụng mô hình đã huấn luyện
    - Tải mô hình
    - Tiền xử lý dữ liệu đầu vào
    - Dự đoán
    - Phát hiện outlier
    """

    def __init__(self, customer_id: str, models_dir: str = None, version: str = 'latest'):
        """
        Khởi tạo ModelPredictor

        Args:
            customer_id: ID của khách hàng
            models_dir: Thư mục chứa mô hình (nếu None, sẽ sử dụng thư mục mặc định)
            version: Phiên bản mô hình ('latest' hoặc phiên bản cụ thể)
        """
        self.customer_id = customer_id
        self.version = version

        # Tải cấu hình
        self.column_config = config_manager.get_column_config(customer_id)
        self.preprocess_config = config_manager.get_preprocessing_config(customer_id)
        self.model_config = config_manager.get_customer_config(customer_id).get('model_config', {})

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

        # Xác định thư mục mô hình
        if models_dir is None:
            self.models_dir = path_manager.get_customer_model_path(customer_id)
        else:
            self.models_dir = models_dir

        # Tải mô hình và metadata
        self.hachtoan_model = None
        self.mahanghua_model = None
        self.outlier_model = None
        self.label_encoders = None
        self.model_metadata = None

        self._load_models()

    # Cập nhật phương thức _load_models trong class ModelPredictor
    def _load_models(self):
        """Tải các mô hình từ đĩa và xử lý các trường hợp đặc biệt"""
        try:
            # Xác định đường dẫn mô hình
            if self.models_dir is None:
                self.models_dir = path_manager.get_customer_model_path(self.customer_id)

            # Tải mô hình HachToan
            hachtoan_path = self._get_model_path('hachtoan')
            if os.path.exists(hachtoan_path):
                try:
                    self.hachtoan_model = joblib.load(hachtoan_path)
                    logger.info(f"Đã tải mô hình HachToan từ: {hachtoan_path}")

                    # Kiểm tra mô hình đã được fit chưa
                    try:
                        if hasattr(self.hachtoan_model, 'steps') and len(self.hachtoan_model.steps) > 0:
                            classifier = self.hachtoan_model.steps[-1][1]
                            if not hasattr(classifier, '_Booster') or classifier._Booster is None:
                                logger.warning("Mô hình HachToan chưa được fit đúng cách - không có _Booster")
                        else:
                            logger.warning("Mô hình HachToan không có bước classifier")
                    except Exception as e:
                        logger.warning(f"Không thể kiểm tra trạng thái fit của mô hình: {str(e)}")
                except Exception as e:
                    logger.error(f"Lỗi khi tải mô hình HachToan: {str(e)}")
                    self.hachtoan_model = None
            else:
                logger.warning(f"Không tìm thấy mô hình HachToan tại: {hachtoan_path}")

            # Tải mô hình MaHangHoa
            mahanghua_path = self._get_model_path('mahanghua')
            if os.path.exists(mahanghua_path):
                try:
                    self.mahanghua_model = joblib.load(mahanghua_path)
                    logger.info(f"Đã tải mô hình MaHangHoa từ: {mahanghua_path}")
                except Exception as e:
                    logger.error(f"Lỗi khi tải mô hình MaHangHoa: {str(e)}")
                    self.mahanghua_model = None
            else:
                logger.info(f"Không tìm thấy mô hình MaHangHoa tại: {mahanghua_path}")

            # Tải mô hình phát hiện outlier
            outlier_path = self._get_model_path('outlier')
            if os.path.exists(outlier_path):
                try:
                    self.outlier_model = joblib.load(outlier_path)
                    logger.info(f"Đã tải mô hình phát hiện outlier từ: {outlier_path}")
                except Exception as e:
                    logger.error(f"Lỗi khi tải mô hình phát hiện outlier: {str(e)}")
                    self.outlier_model = None
            else:
                logger.info(f"Không tìm thấy mô hình phát hiện outlier tại: {outlier_path}")

            # Tải label encoders
            encoders_path = self._get_encoders_path()
            if os.path.exists(encoders_path):
                try:
                    self.label_encoders = joblib.load(encoders_path)
                    logger.info(f"Đã tải label encoders từ: {encoders_path}")
                except Exception as e:
                    logger.error(f"Lỗi khi tải label encoders: {str(e)}")
                    self.label_encoders = {}
            else:
                logger.info(f"Không tìm thấy label encoders tại: {encoders_path}")
                self.label_encoders = {}

            # Tải metadata
            metadata_path = self._get_metadata_path()
            if os.path.exists(metadata_path):
                try:
                    self.model_metadata = load_metadata(metadata_path)
                    logger.info(f"Đã tải metadata từ: {metadata_path}")
                except Exception as e:
                    logger.error(f"Lỗi khi tải metadata: {str(e)}")
                    self.model_metadata = {}
            else:
                logger.warning(f"Không tìm thấy metadata tại: {metadata_path}")
                self.model_metadata = {}

        except Exception as e:
            logger.error(f"Lỗi tổng thể khi tải mô hình: {str(e)}")
            # Khởi tạo các đối tượng trống để tránh lỗi null pointer
            self.hachtoan_model = None
            self.mahanghua_model = None
            self.outlier_model = None
            self.label_encoders = {}
            self.model_metadata = {}

    def _get_model_path(self, model_type: str) -> str:
        """
        Lấy đường dẫn đến file mô hình

        Args:
            model_type: Loại mô hình ('hachtoan', 'mahanghua', hoặc 'outlier')

        Returns:
            Đường dẫn đến file mô hình
        """
        model_dir = os.path.join(self.models_dir, model_type)

        if self.version == 'latest':
            return os.path.join(model_dir, 'model_latest.joblib')
        else:
            return os.path.join(model_dir, f'model_{self.version}.joblib')

    def _get_encoders_path(self) -> str:
        """
        Lấy đường dẫn đến file encoders

        Returns:
            Đường dẫn đến file encoders
        """
        encoders_dir = os.path.join(self.models_dir, 'encoders')

        if self.version == 'latest':
            return os.path.join(encoders_dir, 'label_encoders_latest.joblib')
        else:
            return os.path.join(encoders_dir, f'label_encoders_{self.version}.joblib')

    def _get_metadata_path(self) -> str:
        """
        Lấy đường dẫn đến file metadata

        Returns:
            Đường dẫn đến file metadata
        """
        metadata_dir = os.path.join(self.models_dir, 'metadata')

        if self.version == 'latest':
            return os.path.join(metadata_dir, 'model_metadata_latest.json')
        else:
            return os.path.join(metadata_dir, f'model_metadata_{self.version}.json')

    def _preprocess_input(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu đầu vào cho dự đoán

        Args:
            data: DataFrame hoặc Dict chứa dữ liệu đầu vào

        Returns:
            DataFrame đã được tiền xử lý
        """
        # Chuyển đổi dict thành DataFrame nếu cần
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Kiểm tra các cột bắt buộc
        required_columns = self.vietnamese_text_columns + self.id_columns
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Thiếu các cột bắt buộc: {', '.join(missing_columns)}")

        # Tiền xử lý các cột văn bản tiếng Việt
        for column in self.vietnamese_text_columns:
            if column in df.columns:
                # Xử lý các giá trị null
                df[column] = df[column].fillna("")

                # Chuẩn hóa văn bản
                if self.preprocess_config.get('normalize_text', True):
                    df[column] = df[column].apply(normalize_vietnamese_text)

        # Tiền xử lý các cột ID
        for column in self.id_columns:
            if column in df.columns:
                # Chuyển đổi thành chuỗi
                df[column] = df[column].astype(str)

                # Loại bỏ khoảng trắng thừa
                df[column] = df[column].str.strip()

                # Thay thế giá trị null
                df[column] = df[column].replace('nan', 'unknown')
                df[column] = df[column].fillna('unknown')

        return df

    # Cập nhật phương thức predict trong class ModelPredictor

    def predict(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:
        """
        Dự đoán sử dụng mô hình đã huấn luyện

        Args:
            data: DataFrame hoặc Dict chứa dữ liệu đầu vào

        Returns:
            Dict chứa kết quả dự đoán
        """
        # Khởi tạo kết quả mặc định
        result = {
            "prediction": {},
            "probabilities": {},
            "outlier_warning": False,
            "outlier_score": None,
            "warnings": []
        }

        # Kiểm tra mô hình
        if self.hachtoan_model is None:
            result["warnings"].append("Chưa tải được mô hình HachToan")
            return result

        # Tiền xử lý dữ liệu đầu vào
        try:
            df = self._preprocess_input(data)
        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý dữ liệu đầu vào: {str(e)}")
            result["warnings"].append(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
            return result

        # === 1. Dự đoán HachToan ===
        try:
            # Thử dự đoán thông thường
            try:
                # Kiểm tra xem mô hình đã được fit chưa
                if hasattr(self.hachtoan_model, 'steps') and len(self.hachtoan_model.steps) > 0:
                    classifier = self.hachtoan_model.steps[-1][1]
                    if hasattr(classifier, '_Booster') and classifier._Booster is not None:
                        # Mô hình đã fit, dự đoán bình thường
                        hachtoan_pred = self.hachtoan_model.predict(df)
                        hachtoan_prob = self.hachtoan_model.predict_proba(df)
                    else:
                        # Không thể dự đoán, trả về kết quả mặc định
                        result["warnings"].append("Mô hình HachToan chưa được fit đúng cách")
                        hachtoan_value = "1561"  # Giá trị mặc định hợp lý
                        hachtoan_probability = 0.5
                        result["prediction"]["HachToan"] = str(hachtoan_value)
                        result["probabilities"]["HachToan"] = float(hachtoan_probability)
                        return result
                else:
                    # Không có pipeline đúng, trả về kết quả mặc định
                    result["warnings"].append("Mô hình HachToan không có cấu trúc pipeline hợp lệ")
                    hachtoan_value = "1561"  # Giá trị mặc định hợp lý
                    hachtoan_probability = 0.5
                    result["prediction"]["HachToan"] = str(hachtoan_value)
                    result["probabilities"]["HachToan"] = float(hachtoan_probability)
                    return result

            except Exception as e:
                logger.warning(f"Lỗi khi dự đoán HachToan thông thường: {str(e)}")
                result["warnings"].append(f"Không thể dự đoán HachToan, sử dụng giá trị mặc định")
                # Sử dụng giá trị mặc định thay vì crash
                hachtoan_value = "1561"  # Giá trị mặc định hợp lý
                hachtoan_probability = 0.5
                result["prediction"]["HachToan"] = str(hachtoan_value)
                result["probabilities"]["HachToan"] = float(hachtoan_probability)
                return result

            # Xử lý kết quả dự đoán
            if isinstance(hachtoan_pred, np.ndarray):
                hachtoan_value = hachtoan_pred[0]
            else:
                hachtoan_value = hachtoan_pred

            # Nếu có label encoder, chuyển đổi ngược lại
            if self.label_encoders and 'hachtoan' in self.label_encoders:
                try:
                    hachtoan_value = self.label_encoders['hachtoan'].inverse_transform([hachtoan_value])[0]
                except Exception as e:
                    logger.warning(f"Không thể giải mã HachToan: {str(e)}")
                    # Nếu không thể chuyển đổi, sử dụng giá trị gốc

            # Lấy xác suất cao nhất
            max_prob_idx = np.argmax(hachtoan_prob[0])
            hachtoan_probability = hachtoan_prob[0][max_prob_idx]

            # Thêm vào kết quả
            result["prediction"]["HachToan"] = str(hachtoan_value)
            result["probabilities"]["HachToan"] = float(hachtoan_probability)

            # Cảnh báo nếu xác suất thấp
            if hachtoan_probability < constants.MIN_PROBABILITY_THRESHOLD:
                result["warnings"].append(constants.WARNINGS["low_probability"])

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán HachToan: {str(e)}")
            result["warnings"].append(f"Lỗi dự đoán HachToan: {str(e)}")
            # Không làm gì thêm, chỉ ghi nhận lỗi và tiếp tục

        # === 2. Dự đoán MaHangHoa nếu cần ===
        try:
            hachtoan_value = result["prediction"].get("HachToan", "")
            # Kiểm tra điều kiện
            if (self.starts_with and
                    hachtoan_value and
                    str(hachtoan_value).startswith(self.starts_with) and
                    self.mahanghua_model is not None):

                try:
                    # Thêm cột HachToan vào dữ liệu đầu vào
                    df_with_hachtoan = df.copy()
                    df_with_hachtoan[self.primary_target] = hachtoan_value

                    # Dự đoán MaHangHoa
                    mahanghua_pred = self.mahanghua_model.predict(df_with_hachtoan)
                    mahanghua_prob = self.mahanghua_model.predict_proba(df_with_hachtoan)

                    # Lấy giá trị dự đoán
                    if isinstance(mahanghua_pred, np.ndarray):
                        mahanghua_value = mahanghua_pred[0]
                    else:
                        mahanghua_value = mahanghua_pred

                    # Nếu có label encoder, chuyển đổi ngược lại
                    if 'mahanghua' in self.label_encoders:
                        try:
                            mahanghua_value = self.label_encoders['mahanghua'].inverse_transform([mahanghua_value])[0]
                        except Exception as e:
                            logger.warning(f"Không thể giải mã MaHangHoa: {str(e)}")
                            # Tiếp tục sử dụng giá trị gốc

                    # Lấy xác suất cao nhất
                    max_prob_idx = np.argmax(mahanghua_prob[0])
                    mahanghua_probability = mahanghua_prob[0][max_prob_idx]

                    # Thêm vào kết quả
                    result["prediction"]["MaHangHoa"] = str(mahanghua_value)
                    result["probabilities"]["MaHangHoa"] = float(mahanghua_probability)

                    # Cảnh báo nếu xác suất thấp
                    if mahanghua_probability < constants.MIN_PROBABILITY_THRESHOLD:
                        result["warnings"].append(constants.WARNINGS["low_probability"])
                except Exception as e:
                    logger.warning(f"Lỗi khi dự đoán MaHangHoa: {str(e)}")
                    result["warnings"].append(f"Không thể dự đoán MaHangHoa: {str(e)}")
                    # Sử dụng giá trị mặc định
                    result["prediction"]["MaHangHoa"] = "MH001"
                    result["probabilities"]["MaHangHoa"] = 0.5
            else:
                logger.info("Không thỏa điều kiện để dự đoán MaHangHoa")

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán MaHangHoa: {str(e)}")
            result["warnings"].append(f"Lỗi xử lý MaHangHoa: {str(e)}")
            # Không làm gì thêm, chỉ ghi nhận lỗi và tiếp tục

        # === 3. Phát hiện outlier nếu có mô hình ===
        if self.outlier_model is not None:
            try:
                # Để tránh lỗi "could not convert string to float",
                # chỉ sử dụng các cột số cho outlier detection
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                # Nếu không có cột số nào, thêm một cột giả
                if not numeric_cols:
                    df['_dummy_numeric'] = 0
                    numeric_cols = ['_dummy_numeric']

                # Dữ liệu chỉ có các cột số
                numeric_df = df[numeric_cols].fillna(0)

                # Phát hiện outlier
                outlier_scores = self.outlier_model.decision_function(numeric_df)

                # Chuyển đổi scores thành phạm vi [0, 1]
                # Giá trị càng nhỏ càng có khả năng là outlier
                normalized_score = 1 / (1 + np.exp(-outlier_scores[0]))

                # Lấy ngưỡng từ cấu hình
                threshold = self.model_config.get('outlier_detection', {}).get('threshold', 0.85)

                # Kiểm tra ngưỡng
                is_outlier = normalized_score < threshold

                # Thêm vào kết quả
                result["outlier_warning"] = bool(is_outlier)
                result["outlier_score"] = float(normalized_score)

                # Thêm cảnh báo nếu là outlier
                if is_outlier:
                    result["warnings"].append(constants.WARNINGS["outlier"])

            except Exception as e:
                logger.warning(f"Lỗi khi phát hiện outlier: {str(e)}")
                # Không làm gì thêm, tiếp tục với kết quả hiện tại

        return result

def predict_single_sample(customer_id: str, data: Dict[str, Any], models_dir: str = None,
                          version: str = 'latest') -> Dict[str, Any]:
    """
    Dự đoán cho một mẫu đơn lẻ

    Args:
        customer_id: ID của khách hàng
        data: Dict chứa dữ liệu đầu vào
        models_dir: Thư mục chứa mô hình (nếu None, sẽ sử dụng thư mục mặc định)
        version: Phiên bản mô hình ('latest' hoặc phiên bản cụ thể)

    Returns:
        Dict chứa kết quả dự đoán
    """
    predictor = ModelPredictor(customer_id, models_dir, version)
    return predictor.predict(data)


def predict_batch(customer_id: str, input_file: str, output_file: str = None,
                  models_dir: str = None, version: str = 'latest') -> str:
    """
    Dự đoán hàng loạt từ file CSV

    Args:
        customer_id: ID của khách hàng
        input_file: Đường dẫn đến file CSV đầu vào
        output_file: Đường dẫn đến file CSV đầu ra (nếu None, sẽ tạo tên file tự động)
        models_dir: Thư mục chứa mô hình (nếu None, sẽ sử dụng thư mục mặc định)
        version: Phiên bản mô hình ('latest' hoặc phiên bản cụ thể)

    Returns:
        Đường dẫn đến file kết quả
    """
    logger.info(f"Dự đoán hàng loạt cho khách hàng {customer_id} từ file {input_file}")

    # Đọc dữ liệu đầu vào
    try:
        df = pd.read_csv(input_file, sep=";", encoding='utf-8-sig')
        logger.info(f"Đã đọc {len(df)} dòng từ {input_file}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {input_file}: {str(e)}")
        try:
            df = pd.read_csv(input_file, sep=";", encoding='latin1')
            logger.info(f"Đã đọc {len(df)} dòng từ {input_file} với encoding latin1")
        except Exception as e2:
            logger.error(f"Không thể đọc file {input_file}: {str(e2)}")
            raise ValueError(f"Không thể đọc file đầu vào: {str(e2)}")

    # Tạo predictor
    predictor = ModelPredictor(customer_id, models_dir, version)

    # Tiền xử lý dữ liệu
    processed_df = predictor._preprocess_input(df)

    # Chuẩn bị DataFrame kết quả
    result_df = df.copy()

    # Thêm cột cho kết quả dự đoán
    result_df['Predicted_HachToan'] = None
    result_df['HachToan_Probability'] = None
    result_df['Predicted_MaHangHoa'] = None
    result_df['MaHangHoa_Probability'] = None
    result_df['Is_Outlier'] = False
    result_df['Outlier_Score'] = None
    result_df['Warnings'] = None

    # Dự đoán từng dòng
    logger.info(f"Bắt đầu dự đoán {len(df)} dòng")

    start_time = time.time()
    results = []

    # Xử lý từng dòng
    for idx, row in processed_df.iterrows():
        try:
            # Chuyển dòng thành dict
            row_dict = row.to_dict()

            # Dự đoán
            prediction = predictor.predict(row_dict)

            # Lưu kết quả
            results.append(prediction)

            # Cập nhật DataFrame kết quả
            result_df.at[idx, 'Predicted_HachToan'] = prediction.get('prediction', {}).get('HachToan')
            result_df.at[idx, 'HachToan_Probability'] = prediction.get('probabilities', {}).get('HachToan')

            if 'MaHangHoa' in prediction.get('prediction', {}):
                result_df.at[idx, 'Predicted_MaHangHoa'] = prediction.get('prediction', {}).get('MaHangHoa')
                result_df.at[idx, 'MaHangHoa_Probability'] = prediction.get('probabilities', {}).get('MaHangHoa')

            result_df.at[idx, 'Is_Outlier'] = prediction.get('outlier_warning', False)
            result_df.at[idx, 'Outlier_Score'] = prediction.get('outlier_score')
            result_df.at[idx, 'Warnings'] = '; '.join(prediction.get('warnings', []))

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán dòng {idx}: {str(e)}")
            result_df.at[idx, 'Warnings'] = f"Error: {str(e)}"

    elapsed_time = time.time() - start_time
    logger.info(f"Đã dự đoán xong {len(df)} dòng trong {elapsed_time:.2f} giây")

    # Xác định tên file đầu ra nếu không được cung cấp
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = path_manager.get_customer_data_path(customer_id, 'results')
        output_file = os.path.join(output_dir, f"{customer_id}_predictions_{timestamp}.csv")

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Lưu kết quả
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"Đã lưu kết quả dự đoán vào: {output_file}")

    # Lưu metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "input_file": input_file,
        "output_file": output_file,
        "model_version": version,
        "num_samples": len(df),
        "elapsed_time": elapsed_time,
        "success_rate": (len(df) - result_df['Warnings'].str.contains('Error', na=False).sum()) / len(df)
    }

    metadata_file = os.path.splitext(output_file)[0] + '_metadata.json'
    save_metadata(metadata_file, metadata)

    return output_file


def main():
    """Hàm chính để chạy script từ command line"""
    parser = argparse.ArgumentParser(description='Dự đoán sử dụng mô hình đã huấn luyện')
    parser.add_argument('--customer-id', required=True, help='ID của khách hàng')
    parser.add_argument('--input-file', required=True, help='Đường dẫn đến file CSV đầu vào')
    parser.add_argument('--output-file', help='Đường dẫn đến file CSV đầu ra (tùy chọn)')
    parser.add_argument('--models-dir', help='Thư mục chứa mô hình (tùy chọn)')
    parser.add_argument('--version', default='latest', help='Phiên bản mô hình (mặc định: latest)')

    args = parser.parse_args()

    try:
        output_file = predict_batch(
            customer_id=args.customer_id,
            input_file=args.input_file,
            output_file=args.output_file,
            models_dir=args.models_dir,
            version=args.version
        )

        logger.info(f"Dự đoán hoàn tất, kết quả đã được lưu vào: {output_file}")

        # In ra thông tin để script gọi có thể sử dụng
        print(json.dumps({"output_file": output_file}))

        return 0
    except Exception as e:
        logger.exception(f"Lỗi khi dự đoán: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())