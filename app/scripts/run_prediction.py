#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chạy tác vụ dự đoán hàng loạt hoặc từng mẫu đơn lẻ
- Dự đoán từ file CSV
- Dự đoán từ JSON đầu vào
- Lưu kết quả
"""

import os
import sys
import argparse
import json
import logging
import traceback
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union
import time
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import các module cần thiết
from app.scripts.predict import predict_batch, predict_single_sample
from app.config import config_manager, path_manager
from app.config.utils import save_metadata

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='run_prediction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('run_prediction')


def run_prediction_from_file(customer_id: str, input_file: str, output_file: str = None,
                             models_dir: str = None, version: str = 'latest') -> Dict[str, Any]:
    """
    Chạy dự đoán từ file CSV

    Args:
        customer_id: ID của khách hàng
        input_file: Đường dẫn đến file CSV đầu vào
        output_file: Đường dẫn đến file CSV đầu ra (nếu None, sẽ tạo tên file tự động)
        models_dir: Thư mục chứa mô hình (nếu None, sẽ sử dụng thư mục mặc định)
        version: Phiên bản mô hình ('latest' hoặc phiên bản cụ thể)

    Returns:
        Dict chứa thông tin về kết quả dự đoán
    """
    start_time = time.time()

    logger.info(f"==== BẮT ĐẦU DỰ ĐOÁN CHO KHÁCH HÀNG {customer_id} ====")
    logger.info(f"File đầu vào: {input_file}")

    try:
        # Dự đoán
        result_file = predict_batch(
            customer_id=customer_id,
            input_file=input_file,
            output_file=output_file,
            models_dir=models_dir,
            version=version
        )

        # Tính thời gian
        elapsed_time = time.time() - start_time

        logger.info(f"Dự đoán hoàn tất trong {elapsed_time:.2f} giây")
        logger.info(f"Kết quả đã được lưu vào: {result_file}")

        # Đọc dữ liệu để phân tích
        try:
            result_df = pd.read_csv(result_file, sep=";", encoding='utf-8-sig')

            # Phân tích kết quả
            total_samples = len(result_df)
            outlier_count = result_df['Is_Outlier'].sum() if 'Is_Outlier' in result_df else 0
            outlier_percent = (outlier_count / total_samples * 100) if total_samples > 0 else 0

            warning_count = result_df['Warnings'].notna().sum() if 'Warnings' in result_df else 0
            warning_percent = (warning_count / total_samples * 100) if total_samples > 0 else 0

            # Thống kê dự đoán
            hachtoan_stats = None
            if 'Predicted_HachToan' in result_df:
                hachtoan_counts = result_df['Predicted_HachToan'].value_counts().to_dict()
                hachtoan_stats = {
                    "counts": {str(k): int(v) for k, v in hachtoan_counts.items() if pd.notna(k)},
                    "most_common": str(result_df['Predicted_HachToan'].value_counts().index[0]) if not result_df[
                        'Predicted_HachToan'].value_counts().empty else None
                }

            mahanghua_stats = None
            if 'Predicted_MaHangHoa' in result_df:
                # Lọc ra các dòng có dự đoán MaHangHoa
                mhh_df = result_df[result_df['Predicted_MaHangHoa'].notna()]
                if len(mhh_df) > 0:
                    mahanghua_counts = mhh_df['Predicted_MaHangHoa'].value_counts().to_dict()
                    mahanghua_stats = {
                        "counts": {str(k): int(v) for k, v in mahanghua_counts.items() if pd.notna(k)},
                        "most_common": str(mhh_df['Predicted_MaHangHoa'].value_counts().index[0]) if not mhh_df[
                            'Predicted_MaHangHoa'].value_counts().empty else None,
                        "prediction_count": len(mhh_df),
                        "prediction_percent": (len(mhh_df) / total_samples * 100) if total_samples > 0 else 0
                    }

            # Tạo kết quả đầy đủ
            result = {
                "status": "success",
                "customer_id": customer_id,
                "input_file": input_file,
                "output_file": result_file,
                "model_version": version,
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "total_samples": total_samples,
                "outlier_count": int(outlier_count),
                "outlier_percent": float(outlier_percent),
                "warning_count": int(warning_count),
                "warning_percent": float(warning_percent),
                "hachtoan_stats": hachtoan_stats,
                "mahanghua_stats": mahanghua_stats
            }

        except Exception as e:
            logger.error(f"Lỗi khi phân tích kết quả: {str(e)}")
            result = {
                "status": "partial_success",
                "customer_id": customer_id,
                "input_file": input_file,
                "output_file": result_file,
                "model_version": version,
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "error_analysis": str(e)
            }

        return result

    except Exception as e:
        error_msg = f"Lỗi khi dự đoán: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return {
            "status": "error",
            "customer_id": customer_id,
            "input_file": input_file,
            "model_version": version,
            "timestamp": datetime.now().isoformat(),
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def run_prediction_from_json(customer_id: str, input_data: Dict[str, Any],
                             models_dir: str = None, version: str = 'latest') -> Dict[str, Any]:
    """
    Chạy dự đoán từ dữ liệu JSON

    Args:
        customer_id: ID của khách hàng
        input_data: Dict chứa dữ liệu đầu vào
        models_dir: Thư mục chứa mô hình (nếu None, sẽ sử dụng thư mục mặc định)
        version: Phiên bản mô hình ('latest' hoặc phiên bản cụ thể)

    Returns:
        Dict chứa kết quả dự đoán
    """
    start_time = time.time()

    logger.info(f"==== BẮT ĐẦU DỰ ĐOÁN CHO KHÁCH HÀNG {customer_id} ====")
    logger.info(f"Dữ liệu đầu vào: {json.dumps(input_data)}")

    try:
        # Dự đoán
        prediction = predict_single_sample(
            customer_id=customer_id,
            data=input_data,
            models_dir=models_dir,
            version=version
        )

        # Tính thời gian
        elapsed_time = time.time() - start_time

        logger.info(f"Dự đoán hoàn tất trong {elapsed_time:.2f} giây")

        # Kết quả
        result = {
            "status": "success",
            "customer_id": customer_id,
            "model_version": version,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "prediction": prediction
        }

        return result

    except Exception as e:
        error_msg = f"Lỗi khi dự đoán: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return {
            "status": "error",
            "customer_id": customer_id,
            "model_version": version,
            "timestamp": datetime.now().isoformat(),
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def main():
    """Hàm chính để chạy script từ command line"""
    parser = argparse.ArgumentParser(description='Chạy tác vụ dự đoán')
    parser.add_argument('--customer-id', required=True, help='ID của khách hàng')
    parser.add_argument('--input-file', help='Đường dẫn đến file CSV đầu vào')
    parser.add_argument('--output-file', help='Đường dẫn đến file CSV đầu ra (tùy chọn)')
    parser.add_argument('--input-json', help='Chuỗi JSON đầu vào cho dự đoán đơn lẻ')
    parser.add_argument('--models-dir', help='Thư mục chứa mô hình (tùy chọn)')
    parser.add_argument('--version', default='latest', help='Phiên bản mô hình (mặc định: latest)')
    parser.add_argument('--output-json', help='Đường dẫn đến file JSON kết quả (tùy chọn)')

    args = parser.parse_args()

    # Kiểm tra đối số đầu vào
    if not args.input_file and not args.input_json:
        logger.error("Phải cung cấp một trong hai đối số: --input-file hoặc --input-json")
        return 1

    if args.input_file and args.input_json:
        logger.error("Chỉ được cung cấp một trong hai đối số: --input-file hoặc --input-json")
        return 1

    try:
        # Dự đoán từ file CSV
        if args.input_file:
            result = run_prediction_from_file(
                customer_id=args.customer_id,
                input_file=args.input_file,
                output_file=args.output_file,
                models_dir=args.models_dir,
                version=args.version
            )

        # Dự đoán từ JSON
        else:
            try:
                input_data = json.loads(args.input_json)
            except json.JSONDecodeError:
                logger.error("Chuỗi JSON không hợp lệ")
                return 1

            result = run_prediction_from_json(
                customer_id=args.customer_id,
                input_data=input_data,
                models_dir=args.models_dir,
                version=args.version
            )

        # Lưu kết quả vào file JSON nếu được chỉ định
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu kết quả vào: {args.output_json}")

        # In ra kết quả cho script gọi
        print(json.dumps(result))

        # Trả về mã lỗi
        return 0 if result["status"] == "success" else 1

    except Exception as e:
        logger.exception(f"Lỗi không xử lý được: {str(e)}")
        result = {
            "status": "error",
            "customer_id": args.customer_id,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }

        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    sys.exit(main())