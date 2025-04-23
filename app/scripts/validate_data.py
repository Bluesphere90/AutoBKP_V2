#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script kiểm tra tính hợp lệ của dữ liệu đầu vào
- Kiểm tra cấu trúc file CSV
- Kiểm tra tính đầy đủ của dữ liệu
- Báo cáo các vấn đề tiềm ẩn
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import config_manager, path_manager, constants
from app.config.utils import detect_vietnamese_columns, is_vietnamese_text

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='validate_data.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('validate_data')


def read_csv_file(file_path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Đọc file CSV và trả về DataFrame

    Args:
        file_path: Đường dẫn đến file CSV

    Returns:
        Tuple (DataFrame, encoding_used)
    """
    # Danh sách các encoding thường gặp
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, sep=";", encoding=encoding)
            logger.info(f"Đã đọc file CSV với encoding {encoding}")
            return df, encoding
        except UnicodeDecodeError:
            logger.debug(f"Không thể đọc với encoding {encoding}, thử encoding khác")
        except Exception as e:
            logger.error(f"Lỗi khi đọc file: {str(e)}")
            raise

    # Nếu không đọc được với bất kỳ encoding nào
    raise ValueError(f"Không thể đọc file CSV với các encoding đã thử: {encodings}")


def validate_column_presence(df: pd.DataFrame, customer_id: str) -> Dict[str, Any]:
    """
    Kiểm tra sự hiện diện của các cột yêu cầu

    Args:
        df: DataFrame cần kiểm tra
        customer_id: ID của khách hàng

    Returns:
        Dict chứa kết quả kiểm tra
    """
    column_config = config_manager.get_column_config(customer_id)

    # Lấy danh sách các cột bắt buộc
    required_columns = []

    # Cột ID
    id_columns = column_config.get('id_columns', [])
    required_columns.extend(id_columns)

    # Cột văn bản tiếng Việt
    vietnamese_text_columns = column_config.get('vietnamese_text_columns', [])
    required_columns.extend(vietnamese_text_columns)

    # Cột mục tiêu chính
    primary_target = column_config.get('target_columns', {}).get('primary')
    if primary_target:
        required_columns.append(primary_target)

    # Cột mục tiêu thứ cấp (không bắt buộc cho huấn luyện ban đầu)
    secondary_target = column_config.get('target_columns', {}).get('secondary')

    # Kiểm tra các cột hiện có
    found_columns = []
    missing_columns = []

    for col in required_columns:
        if col in df.columns:
            found_columns.append(col)
        else:
            missing_columns.append(col)

    # Kiểm tra cột mục tiêu thứ cấp
    has_secondary_target = secondary_target in df.columns if secondary_target else False

    result = {
        "required_columns": required_columns,
        "found_columns": found_columns,
        "missing_columns": missing_columns,
        "has_all_required": len(missing_columns) == 0,
        "has_secondary_target": has_secondary_target,
        "all_columns": list(df.columns)
    }

    return result


def validate_data_completeness(df: pd.DataFrame, customer_id: str) -> Dict[str, Any]:
    """
    Kiểm tra tính đầy đủ của dữ liệu

    Args:
        df: DataFrame cần kiểm tra
        customer_id: ID của khách hàng

    Returns:
        Dict chứa kết quả kiểm tra
    """
    column_config = config_manager.get_column_config(customer_id)

    # Lấy danh sách các cột cần kiểm tra
    columns_to_check = []

    # Cột ID
    columns_to_check.extend(column_config.get('id_columns', []))

    # Cột văn bản tiếng Việt
    columns_to_check.extend(column_config.get('vietnamese_text_columns', []))

    # Cột mục tiêu
    primary_target = column_config.get('target_columns', {}).get('primary')
    if primary_target:
        columns_to_check.append(primary_target)

    # Kết quả kiểm tra
    completeness = {}
    problematic_columns = []

    # Kiểm tra từng cột
    for col in columns_to_check:
        if col in df.columns:
            # Số lượng giá trị null
            null_count = df[col].isnull().sum()
            null_percent = (null_count / len(df)) * 100

            # Số lượng giá trị rỗng
            empty_count = (df[col] == '').sum() if df[col].dtype == 'object' else 0
            empty_percent = (empty_count / len(df)) * 100

            # Tổng số giá trị trống
            total_missing = null_count + empty_count
            total_missing_percent = (total_missing / len(df)) * 100

            # Lưu kết quả
            completeness[col] = {
                "null_count": int(null_count),
                "null_percent": float(null_percent),
                "empty_count": int(empty_count),
                "empty_percent": float(empty_percent),
                "total_missing": int(total_missing),
                "total_missing_percent": float(total_missing_percent)
            }

            # Đánh dấu cột có vấn đề
            if total_missing_percent > 5:  # Ngưỡng 5%
                problematic_columns.append({
                    "column": col,
                    "missing_percent": total_missing_percent,
                    "severity": "high" if total_missing_percent > 20 else "medium"
                })

    result = {
        "completeness": completeness,
        "problematic_columns": problematic_columns,
        "has_completeness_issues": len(problematic_columns) > 0
    }

    return result


def validate_data_types(df: pd.DataFrame, customer_id: str) -> Dict[str, Any]:
    """
    Kiểm tra kiểu dữ liệu của các cột

    Args:
        df: DataFrame cần kiểm tra
        customer_id: ID của khách hàng

    Returns:
        Dict chứa kết quả kiểm tra
    """
    column_config = config_manager.get_column_config(customer_id)

    # Lấy thông tin các cột
    data_types = {}
    type_issues = []

    # Kiểm tra từng cột
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        unique_percent = (unique_count / len(df)) * 100

        # Phân loại cột
        col_type = "unknown"

        if col in column_config.get('id_columns', []):
            col_type = "id"
            # ID thường không nên có quá nhiều giá trị null hoặc quá ít giá trị unique
            if df[col].isnull().sum() > 0:
                type_issues.append({
                    "column": col,
                    "issue": "ID column contains null values",
                    "severity": "medium"
                })

        elif col in column_config.get('vietnamese_text_columns', []):
            col_type = "vietnamese_text"
            # Văn bản tiếng Việt nên có kiểu object
            if dtype != 'object':
                type_issues.append({
                    "column": col,
                    "issue": f"Vietnamese text column has non-object data type: {dtype}",
                    "severity": "high"
                })

            # Kiểm tra xem có thực sự chứa văn bản tiếng Việt không
            sample = df[col].dropna().astype(str).sample(min(100, len(df))).tolist()
            vietnamese_count = sum(1 for text in sample if is_vietnamese_text(text))
            vietnamese_percent = (vietnamese_count / len(sample)) * 100

            if vietnamese_percent < 10:  # Ngưỡng 10%
                type_issues.append({
                    "column": col,
                    "issue": f"Column may not contain Vietnamese text (only {vietnamese_percent:.1f}% detected)",
                    "severity": "medium"
                })

        elif col == column_config.get('target_columns', {}).get('primary'):
            col_type = "primary_target"
            # Mục tiêu nên có nhiều giá trị unique
            if unique_count < 2:
                type_issues.append({
                    "column": col,
                    "issue": "Primary target has less than 2 unique values",
                    "severity": "high"
                })

        elif col == column_config.get('target_columns', {}).get('secondary'):
            col_type = "secondary_target"
            # Mục tiêu nên có nhiều giá trị unique
            if unique_count < 2:
                type_issues.append({
                    "column": col,
                    "issue": "Secondary target has less than 2 unique values",
                    "severity": "high"
                })

        # Lưu thông tin
        data_types[col] = {
            "dtype": str(dtype),
            "unique_count": int(unique_count),
            "unique_percent": float(unique_percent),
            "inferred_type": str(pd.api.types.infer_dtype(df[col])),
            "column_type": col_type
        }

    result = {
        "data_types": data_types,
        "type_issues": type_issues,
        "has_type_issues": len(type_issues) > 0
    }

    return result


def validate_target_distribution(df: pd.DataFrame, customer_id: str) -> Dict[str, Any]:
    """
    Kiểm tra phân phối của biến mục tiêu

    Args:
        df: DataFrame cần kiểm tra
        customer_id: ID của khách hàng

    Returns:
        Dict chứa kết quả kiểm tra
    """
    column_config = config_manager.get_column_config(customer_id)

    # Lấy thông tin biến mục tiêu
    primary_target = column_config.get('target_columns', {}).get('primary')
    secondary_target = column_config.get('target_columns', {}).get('secondary')
    condition_column = column_config.get('target_columns', {}).get('secondary_condition', {}).get('column')
    starts_with = column_config.get('target_columns', {}).get('secondary_condition', {}).get('starts_with')

    result = {}

    # Kiểm tra biến mục tiêu chính
    if primary_target and primary_target in df.columns:
        # Đếm số lượng mẫu cho mỗi lớp
        value_counts = df[primary_target].value_counts()
        class_counts = value_counts.to_dict()
        total_samples = len(df)
        num_classes = len(class_counts)

        # Tính tỷ lệ cho mỗi lớp
        class_ratios = {str(cls): count / total_samples for cls, count in class_counts.items()}

        # Tính các chỉ số về mất cân bằng
        if len(class_counts) >= 2:
            min_class_count = value_counts.min()
            max_class_count = value_counts.max()
            imbalance_ratio = max_class_count / min_class_count
            is_imbalanced = imbalance_ratio > constants.IMBALANCE_THRESHOLD
            is_severely_imbalanced = imbalance_ratio > constants.SEVERE_IMBALANCE_THRESHOLD
        else:
            min_class_count = max_class_count = value_counts.iloc[0] if not value_counts.empty else 0
            imbalance_ratio = 1.0
            is_imbalanced = False
            is_severely_imbalanced = False

        # Tìm các lớp hiếm
        minority_threshold = 0.01  # 1%
        minority_classes = {str(cls): ratio for cls, ratio in class_ratios.items() if ratio < minority_threshold}

        # Lưu kết quả
        result["primary_target"] = {
            "column": primary_target,
            "num_classes": num_classes,
            "min_class_count": int(min_class_count),
            "max_class_count": int(max_class_count),
            "imbalance_ratio": float(imbalance_ratio),
            "is_imbalanced": is_imbalanced,
            "is_severely_imbalanced": is_severely_imbalanced,
            "minority_classes": minority_classes,
            "top_classes": {str(k): int(v) for k, v in
                            sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]},
            "class_ratios": {str(k): float(v) for k, v in
                             sorted(class_ratios.items(), key=lambda x: x[1], reverse=True)[:10]}
        }

    # Kiểm tra biến mục tiêu thứ cấp
    if secondary_target and secondary_target in df.columns:
        # Lọc dữ liệu theo điều kiện nếu có
        filtered_df = df
        if condition_column and starts_with and condition_column in df.columns:
            condition_mask = df[condition_column].astype(str).str.startswith(starts_with)
            filtered_df = df[condition_mask]

            # Số lượng mẫu thỏa điều kiện
            condition_count = condition_mask.sum()
            condition_percent = (condition_count / len(df)) * 100
        else:
            condition_count = len(df)
            condition_percent = 100.0

        # Đếm số lượng mẫu cho mỗi lớp
        if len(filtered_df) > 0:
            value_counts = filtered_df[secondary_target].value_counts()
            class_counts = value_counts.to_dict()
            total_samples = len(filtered_df)
            num_classes = len(class_counts)

            # Tính tỷ lệ cho mỗi lớp
            class_ratios = {str(cls): count / total_samples for cls, count in class_counts.items()}

            # Tính các chỉ số về mất cân bằng
            if len(class_counts) >= 2:
                min_class_count = value_counts.min()
                max_class_count = value_counts.max()
                imbalance_ratio = max_class_count / min_class_count
                is_imbalanced = imbalance_ratio > constants.IMBALANCE_THRESHOLD
                is_severely_imbalanced = imbalance_ratio > constants.SEVERE_IMBALANCE_THRESHOLD
            else:
                min_class_count = max_class_count = value_counts.iloc[0] if not value_counts.empty else 0
                imbalance_ratio = 1.0
                is_imbalanced = False
                is_severely_imbalanced = False

            # Tìm các lớp hiếm
            minority_threshold = 0.01  # 1%
            minority_classes = {str(cls): ratio for cls, ratio in class_ratios.items() if ratio < minority_threshold}

            # Lưu kết quả
            result["secondary_target"] = {
                "column": secondary_target,
                "condition_column": condition_column,
                "condition_value": starts_with,
                "condition_count": int(condition_count),
                "condition_percent": float(condition_percent),
                "num_classes": num_classes,
                "min_class_count": int(min_class_count),
                "max_class_count": int(max_class_count),
                "imbalance_ratio": float(imbalance_ratio),
                "is_imbalanced": is_imbalanced,
                "is_severely_imbalanced": is_severely_imbalanced,
                "minority_classes": minority_classes,
                "top_classes": {str(k): int(v) for k, v in
                                sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]},
                "class_ratios": {str(k): float(v) for k, v in
                                 sorted(class_ratios.items(), key=lambda x: x[1], reverse=True)[:10]}
            }
        else:
            # Không có mẫu thỏa điều kiện
            result["secondary_target"] = {
                "column": secondary_target,
                "condition_column": condition_column,
                "condition_value": starts_with,
                "condition_count": 0,
                "condition_percent": 0.0,
                "warning": "No samples match the condition for secondary target"
            }

            # Cảnh báo về mất cân bằng dữ liệu
        imbalance_warnings = []

        if "primary_target" in result and result["primary_target"].get("is_severely_imbalanced", False):
            imbalance_warnings.append({
                "target": "primary",
                "imbalance_ratio": result["primary_target"]["imbalance_ratio"],
                "warning": "Severe class imbalance detected in primary target",
                "severity": "high"
            })
        elif "primary_target" in result and result["primary_target"].get("is_imbalanced", False):
            imbalance_warnings.append({
                "target": "primary",
                "imbalance_ratio": result["primary_target"]["imbalance_ratio"],
                "warning": "Class imbalance detected in primary target",
                "severity": "medium"
            })

        if "secondary_target" in result and result["secondary_target"].get("is_severely_imbalanced", False):
            imbalance_warnings.append({
                "target": "secondary",
                "imbalance_ratio": result["secondary_target"]["imbalance_ratio"],
                "warning": "Severe class imbalance detected in secondary target",
                "severity": "high"
            })
        elif "secondary_target" in result and result["secondary_target"].get("is_imbalanced", False):
            imbalance_warnings.append({
                "target": "secondary",
                "imbalance_ratio": result["secondary_target"]["imbalance_ratio"],
                "warning": "Class imbalance detected in secondary target",
                "severity": "medium"
            })

        result["imbalance_warnings"] = imbalance_warnings
        result["has_imbalance_issues"] = len(imbalance_warnings) > 0

        return result

    def validate_data(file_path: str, customer_id: str) -> Dict[str, Any]:
        """
        Thực hiện toàn bộ kiểm tra tính hợp lệ của dữ liệu

        Args:
            file_path: Đường dẫn đến file CSV
            customer_id: ID của khách hàng

        Returns:
            Dict chứa kết quả kiểm tra
        """
        logger.info(f"Bắt đầu kiểm tra tính hợp lệ của dữ liệu từ file {file_path}")

        # Đọc file CSV
        try:
            df, encoding = read_csv_file(file_path)
        except Exception as e:
            logger.error(f"Không thể đọc file: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path,
                "customer_id": customer_id,
                "timestamp": datetime.now().isoformat()
            }

        # Thông tin cơ bản về file
        file_info = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "encoding": encoding,
            "num_rows": len(df),
            "num_columns": len(df.columns)
        }

        # Thực hiện các kiểm tra
        column_presence = validate_column_presence(df, customer_id)
        data_completeness = validate_data_completeness(df, customer_id)
        data_types = validate_data_types(df, customer_id)
        target_distribution = validate_target_distribution(df, customer_id)

        # Tổng hợp các cảnh báo
        warnings = []

        # Cảnh báo về cột thiếu
        if not column_presence["has_all_required"]:
            warnings.append({
                "category": "missing_columns",
                "message": f"Missing required columns: {', '.join(column_presence['missing_columns'])}",
                "severity": "high"
            })

        # Cảnh báo về dữ liệu không đầy đủ
        for col_issue in data_completeness.get("problematic_columns", []):
            warnings.append({
                "category": "data_completeness",
                "message": f"Column '{col_issue['column']}' has {col_issue['missing_percent']:.1f}% missing values",
                "severity": col_issue["severity"]
            })

        # Cảnh báo về kiểu dữ liệu
        for type_issue in data_types.get("type_issues", []):
            warnings.append({
                "category": "data_types",
                "message": type_issue["issue"],
                "severity": type_issue["severity"]
            })

        # Cảnh báo về mất cân bằng dữ liệu
        for imbalance_warning in target_distribution.get("imbalance_warnings", []):
            warnings.append({
                "category": "class_imbalance",
                "message": imbalance_warning["warning"],
                "severity": imbalance_warning["severity"]
            })

        # Cảnh báo về kích thước dữ liệu
        if len(df) < 100:
            warnings.append({
                "category": "data_size",
                "message": f"Dataset is very small ({len(df)} samples). Model performance may be limited.",
                "severity": "medium"
            })

        # Đánh giá tổng thể
        if len([w for w in warnings if w["severity"] == "high"]) > 0:
            overall_status = "critical_issues"
        elif len(warnings) > 0:
            overall_status = "warnings"
        else:
            overall_status = "valid"

        # Kết quả tổng hợp
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "file_info": file_info,
            "column_presence": column_presence,
            "data_completeness": data_completeness,
            "data_types": data_types,
            "target_distribution": target_distribution,
            "warnings": warnings,
            "overall_status": overall_status
        }

        logger.info(f"Hoàn tất kiểm tra. Trạng thái: {overall_status}")
        logger.info(f"Tổng số cảnh báo: {len(warnings)}")

        return result

    def main():
        """Hàm chính để chạy script từ command line"""
        parser = argparse.ArgumentParser(description='Kiểm tra tính hợp lệ của dữ liệu')
        parser.add_argument('--file-path', required=True, help='Đường dẫn đến file CSV cần kiểm tra')
        parser.add_argument('--customer-id', required=True, help='ID của khách hàng')
        parser.add_argument('--output-file', help='Đường dẫn đến file JSON kết quả (tùy chọn)')

        args = parser.parse_args()

        try:
            result = validate_data(args.file_path, args.customer_id)

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
            result = {
                "status": "error",
                "error": str(e),
                "file_path": args.file_path,
                "customer_id": args.customer_id,
                "timestamp": datetime.now().isoformat()
            }

            print(json.dumps(result))
            return 1

    if __name__ == "__main__":
        sys.exit(main())