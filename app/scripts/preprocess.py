#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script tiền xử lý dữ liệu
- Đọc dữ liệu từ CSV
- Xử lý các cột văn bản tiếng Việt
- Xử lý các cột ID
- Chuẩn bị dữ liệu huấn luyện
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import config_manager, path_manager, constants
from app.config.utils import detect_vietnamese_columns, normalize_vietnamese_text, save_metadata

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='preprocess.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('preprocess')


def read_input_data(file_path: str) -> pd.DataFrame:
    """
    Đọc dữ liệu đầu vào từ file CSV

    Args:
        file_path: Đường dẫn đến file CSV

    Returns:
        DataFrame chứa dữ liệu
    """
    logger.info(f"Đọc dữ liệu từ: {file_path}")

    try:
        # Thử đọc với các encoding khác nhau
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, sep=";", encoding=encoding)
                logger.info(f"Đọc thành công với encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue

        # Nếu không đọc được với bất kỳ encoding nào
        raise ValueError(f"Không thể đọc file CSV với các encoding đã thử: {encodings}")

    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        raise


def validate_data(df: pd.DataFrame, column_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Kiểm tra tính hợp lệ của dữ liệu

    Args:
        df: DataFrame cần kiểm tra
        column_config: Cấu hình cột từ file config

    Returns:
        Tuple (is_valid, error_messages)
    """
    errors = []

    # Kiểm tra các cột bắt buộc
    required_columns = (
            column_config.get('id_columns', []) +
            column_config.get('vietnamese_text_columns', []) +
            [column_config.get('target_columns', {}).get('primary', '')]
    )

    missing_columns = [col for col in required_columns if col and col not in df.columns]

    if missing_columns:
        errors.append(f"Thiếu các cột bắt buộc: {', '.join(missing_columns)}")

    # Kiểm tra số lượng dữ liệu
    if len(df) < 10:
        errors.append(f"Không đủ dữ liệu để xử lý: chỉ có {len(df)} dòng")

    # Kiểm tra giá trị null trong các cột bắt buộc
    for col in required_columns:
        if col and col in df.columns and df[col].isnull().sum() > 0:
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df)) * 100
            errors.append(f"Cột {col} có {null_count} giá trị null ({null_percent:.2f}%)")

    # Kiểm tra mục tiêu thứ cấp nếu cần
    secondary_target = column_config.get('target_columns', {}).get('secondary')
    if secondary_target and secondary_target in df.columns:
        # Kiểm tra điều kiện để dự đoán mục tiêu thứ cấp
        condition_column = column_config.get('target_columns', {}).get('secondary_condition', {}).get('column')
        starts_with = column_config.get('target_columns', {}).get('secondary_condition', {}).get('starts_with')

        if condition_column and starts_with:
            condition_mask = df[condition_column].astype(str).str.startswith(starts_with)
            if condition_mask.sum() == 0:
                errors.append(f"Không có dữ liệu nào thỏa điều kiện {condition_column}.startswith('{starts_with}')")

    return len(errors) == 0, errors


def preprocess_text_column(df: pd.DataFrame, column: str,
                           preprocess_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Tiền xử lý cột văn bản tiếng Việt

    Args:
        df: DataFrame chứa dữ liệu
        column: Tên cột cần xử lý
        preprocess_config: Cấu hình tiền xử lý

    Returns:
        DataFrame đã được xử lý
    """
    logger.info(f"Tiền xử lý cột văn bản tiếng Việt: {column}")

    # Tạo một bản sao để tránh cảnh báo SettingWithCopyWarning
    df = df.copy()

    # Xử lý các giá trị null
    df[column] = df[column].fillna("")

    # Chuẩn hóa văn bản
    if preprocess_config.get('normalize_text', True):
        df[column] = df[column].apply(normalize_vietnamese_text)

    # Loại bỏ các dòng có văn bản quá ngắn
    min_length = preprocess_config.get('min_text_length', 1)
    if min_length > 1:
        mask = df[column].str.len() >= min_length
        df = df[mask].reset_index(drop=True)
        logger.info(f"Đã loại bỏ {(~mask).sum()} dòng có văn bản quá ngắn (<{min_length} ký tự)")

    return df


def preprocess_id_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Tiền xử lý cột ID

    Args:
        df: DataFrame chứa dữ liệu
        column: Tên cột cần xử lý

    Returns:
        DataFrame đã được xử lý
    """
    logger.info(f"Tiền xử lý cột ID: {column}")

    # Tạo một bản sao để tránh cảnh báo SettingWithCopyWarning
    df = df.copy()

    # Chuyển đổi thành chuỗi
    df[column] = df[column].astype(str)

    # Loại bỏ khoảng trắng thừa
    df[column] = df[column].str.strip()

    # Thay thế giá trị null
    df[column] = df[column].replace('nan', 'unknown')
    df[column] = df[column].fillna('unknown')

    return df


def analyze_target_distribution(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Phân tích phân phối của biến mục tiêu

    Args:
        df: DataFrame chứa dữ liệu
        target_column: Tên cột mục tiêu

    Returns:
        Dict chứa thông tin phân tích
    """
    logger.info(f"Phân tích phân phối của biến mục tiêu: {target_column}")

    # Đếm số lượng mẫu cho mỗi lớp
    class_counts = df[target_column].value_counts().to_dict()
    total_samples = len(df)

    # Tính tỷ lệ cho mỗi lớp
    class_ratios = {cls: count / total_samples for cls, count in class_counts.items()}

    # Tính các chỉ số về mất cân bằng
    num_classes = len(class_counts)
    min_class_count = min(class_counts.values()) if class_counts else 0
    max_class_count = max(class_counts.values()) if class_counts else 0

    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

    # Xác định các lớp thiểu số và lớp đa số
    minority_threshold = 0.05  # 5%
    majority_threshold = 0.2  # 20%

    minority_classes = {cls: ratio for cls, ratio in class_ratios.items() if ratio < minority_threshold}
    majority_classes = {cls: ratio for cls, ratio in class_ratios.items() if ratio > majority_threshold}

    analysis = {
        "num_classes": num_classes,
        "total_samples": total_samples,
        "class_counts": class_counts,
        "class_ratios": class_ratios,
        "imbalance_ratio": imbalance_ratio,
        "min_class_count": min_class_count,
        "max_class_count": max_class_count,
        "minority_classes": minority_classes,
        "majority_classes": majority_classes,
        "is_imbalanced": imbalance_ratio > constants.IMBALANCE_THRESHOLD,
        "is_severely_imbalanced": imbalance_ratio > constants.SEVERE_IMBALANCE_THRESHOLD
    }

    logger.info(f"Số lượng lớp: {num_classes}")
    logger.info(f"Tỷ lệ mất cân bằng: {imbalance_ratio:.2f}")
    logger.info(f"Lớp hiếm nhất có {min_class_count} mẫu ({min(class_ratios.values()) * 100:.2f}%)")
    logger.info(f"Lớp phổ biến nhất có {max_class_count} mẫu ({max(class_ratios.values()) * 100:.2f}%)")

    return analysis


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra

    Args:
        df: DataFrame chứa dữ liệu
        test_size: Tỷ lệ dữ liệu dành cho tập kiểm tra
        random_state: Hạt giống ngẫu nhiên

    Returns:
        Tuple (train_df, test_df)
    """
    logger.info(f"Phân chia dữ liệu với test_size={test_size}")

    # Đảm bảo dữ liệu được xáo trộn
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Tính số lượng mẫu cho tập huấn luyện
    train_size = int(len(df) * (1 - test_size))

    # Phân chia dữ liệu
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    logger.info(f"Tập huấn luyện: {len(train_df)} mẫu")
    logger.info(f"Tập kiểm tra: {len(test_df)} mẫu")

    return train_df, test_df


def prepare_data_for_training(customer_id: str, input_file: str, output_dir: str = None) -> Dict[str, str]:
    """
    Chuẩn bị dữ liệu cho quá trình huấn luyện

    Args:
        customer_id: ID của khách hàng
        input_file: Đường dẫn đến file CSV đầu vào
        output_dir: Thư mục đầu ra (nếu None, sẽ sử dụng thư mục mặc định)

    Returns:
        Dict chứa đường dẫn đến các file dữ liệu đã xử lý
    """
    logger.info(f"Bắt đầu tiền xử lý dữ liệu cho khách hàng {customer_id}")

    # Tải cấu hình
    column_config = config_manager.get_column_config(customer_id)
    preprocess_config = config_manager.get_preprocessing_config(customer_id)
    training_config = config_manager.get_training_config(customer_id)

    # Đọc dữ liệu
    df = read_input_data(input_file)
    logger.info(f"Đã đọc {len(df)} dòng dữ liệu từ {input_file}")

    # Kiểm tra tính hợp lệ của dữ liệu
    is_valid, errors = validate_data(df, column_config)
    if not is_valid:
        for error in errors:
            logger.error(error)
        logger.error("Dữ liệu không hợp lệ, dừng xử lý")
        raise ValueError(f"Dữ liệu không hợp lệ: {errors}")

    # Xác định các cột chứa văn bản tiếng Việt nếu không được chỉ định
    vietnamese_text_columns = column_config.get('vietnamese_text_columns', [])
    if not vietnamese_text_columns:
        vietnamese_text_columns = detect_vietnamese_columns(df)
        logger.info(f"Đã phát hiện các cột chứa văn bản tiếng Việt: {vietnamese_text_columns}")

    # Tiền xử lý các cột văn bản tiếng Việt
    for column in vietnamese_text_columns:
        if column in df.columns:
            df = preprocess_text_column(df, column, preprocess_config)

    # Tiền xử lý các cột ID
    id_columns = column_config.get('id_columns', [])
    for column in id_columns:
        if column in df.columns:
            df = preprocess_id_column(df, column)

    # Lấy tên các cột mục tiêu
    primary_target = column_config.get('target_columns', {}).get('primary')
    secondary_target = column_config.get('target_columns', {}).get('secondary')

    # Phân tích phân phối của biến mục tiêu chính
    primary_target_analysis = analyze_target_distribution(df, primary_target)

    # Phân tích phân phối của biến mục tiêu thứ cấp (nếu có)
    secondary_target_analysis = None
    if secondary_target and secondary_target in df.columns:
        # Lấy điều kiện
        condition_column = column_config.get('target_columns', {}).get('secondary_condition', {}).get('column')
        starts_with = column_config.get('target_columns', {}).get('secondary_condition', {}).get('starts_with')

        if condition_column and starts_with:
            # Lọc dữ liệu theo điều kiện
            condition_mask = df[condition_column].astype(str).str.startswith(starts_with)
            if condition_mask.sum() > 0:
                secondary_df = df[condition_mask]
                secondary_target_analysis = analyze_target_distribution(secondary_df, secondary_target)

    # Phân chia dữ liệu
    train_df, test_df = split_data(
        df,
        test_size=training_config.get('test_size', 0.2),
        random_state=training_config.get('random_state', 42)
    )

    # Xác định thư mục đầu ra
    if output_dir is None:
        output_dir = path_manager.get_customer_data_path(customer_id, 'processed')

    os.makedirs(output_dir, exist_ok=True)

    # Tạo timestamp cho tên file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Lưu dữ liệu đã xử lý
    train_file = os.path.join(output_dir, f"{customer_id}_train_{timestamp}.csv")
    test_file = os.path.join(output_dir, f"{customer_id}_test_{timestamp}.csv")

    train_df.to_csv(train_file, index=False, encoding='utf-8-sig', sep=";")
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig', sep=";")

    logger.info(f"Đã lưu tập huấn luyện: {train_file}")
    logger.info(f"Đã lưu tập kiểm tra: {test_file}")

    # Lưu metadata
    metadata = {
        "timestamp": timestamp,
        "customer_id": customer_id,
        "original_file": input_file,
        "train_file": train_file,
        "test_file": test_file,
        "num_samples": len(df),
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df),
        "columns": list(df.columns),
        "vietnamese_text_columns": vietnamese_text_columns,
        "id_columns": id_columns,
        "primary_target": primary_target,
        "primary_target_analysis": primary_target_analysis,
        "secondary_target": secondary_target,
        "secondary_target_analysis": secondary_target_analysis,
        "column_config": column_config,
        "preprocess_config": preprocess_config
    }

    metadata_file = os.path.join(output_dir, f"{customer_id}_metadata_{timestamp}.json")
    save_metadata(metadata_file, metadata)
    logger.info(f"Đã lưu metadata: {metadata_file}")

    return {
        "train_file": train_file,
        "test_file": test_file,
        "metadata_file": metadata_file
    }


def main():
    """Hàm chính để chạy script từ command line"""
    parser = argparse.ArgumentParser(description='Tiền xử lý dữ liệu cho huấn luyện mô hình')
    parser.add_argument('--customer-id', required=True, help='ID của khách hàng')
    parser.add_argument('--input-file', required=True, help='Đường dẫn đến file CSV đầu vào')
    parser.add_argument('--output-dir', help='Thư mục đầu ra (tùy chọn)')

    args = parser.parse_args()

    try:
        result = prepare_data_for_training(
            customer_id=args.customer_id,
            input_file=args.input_file,
            output_dir=args.output_dir
        )

        logger.info("Tiền xử lý dữ liệu hoàn tất")
        logger.info(f"Tập huấn luyện: {result['train_file']}")
        logger.info(f"Tập kiểm tra: {result['test_file']}")
        logger.info(f"Metadata: {result['metadata_file']}")

        # In ra thông tin để script gọi có thể sử dụng
        print(json.dumps(result))

        return 0
    except Exception as e:
        logger.exception(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())