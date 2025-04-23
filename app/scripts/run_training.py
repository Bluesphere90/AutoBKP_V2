#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chạy toàn bộ quy trình huấn luyện từ đầu đến cuối:
1. Tiền xử lý dữ liệu
2. Huấn luyện mô hình
3. Đánh giá kết quả
"""

import os
import sys
import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any
import time
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import các module cần thiết
from app.scripts.preprocess import prepare_data_for_training
# Import toàn bộ module train để tránh vấn đề tham chiếu
import app.scripts.train as train_module
from app.config import config_manager, path_manager

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='run_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('run_training')


def run_complete_training(customer_id: str, input_file: str) -> Dict[str, Any]:
    """
    Chạy toàn bộ quy trình huấn luyện

    Args:
        customer_id: ID của khách hàng
        input_file: Đường dẫn đến file CSV dữ liệu gốc

    Returns:
        Dict chứa thông tin về các file đã tạo
    """
    start_time = time.time()

    logger.info(f"==== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN CHO KHÁCH HÀNG {customer_id} ====")
    logger.info(f"File dữ liệu đầu vào: {input_file}")

    results = {}

    try:
        # 1. Tiền xử lý dữ liệu
        logger.info("1. Tiền xử lý dữ liệu")
        preprocess_start = time.time()

        preprocess_result = prepare_data_for_training(
            customer_id=customer_id,
            input_file=input_file
        )

        preprocess_time = time.time() - preprocess_start
        logger.info(f"Tiền xử lý dữ liệu hoàn tất trong {preprocess_time:.2f} giây")
        logger.info(f"Tập huấn luyện: {preprocess_result['train_file']}")
        logger.info(f"Tập kiểm tra: {preprocess_result['test_file']}")

        results['preprocess'] = preprocess_result
        results['preprocess']['time'] = preprocess_time

        # 2. Huấn luyện mô hình
        logger.info("2. Huấn luyện mô hình")
        training_start = time.time()

        training_result = train_module.train_customer_model(
            customer_id=customer_id,
            train_file=preprocess_result['train_file'],
            test_file=preprocess_result['test_file']
        )

        training_time = time.time() - training_start
        logger.info(f"Huấn luyện mô hình hoàn tất trong {training_time:.2f} giây")

        results['training'] = training_result
        results['training']['time'] = training_time

        # 3. Tổng kết
        total_time = time.time() - start_time
        logger.info(f"==== QUY TRÌNH HUẤN LUYỆN HOÀN TẤT TRONG {total_time:.2f} GIÂY ====")

        results['total_time'] = total_time
        results['status'] = 'success'
        results['timestamp'] = datetime.now().isoformat()

        return results

    except Exception as e:
        error_msg = f"Lỗi khi chạy quy trình huấn luyện: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        results['status'] = 'error'
        results['error'] = error_msg
        results['traceback'] = traceback.format_exc()
        results['timestamp'] = datetime.now().isoformat()

        return results


def main():
    """Hàm chính để chạy script từ command line"""
    parser = argparse.ArgumentParser(description='Chạy toàn bộ quy trình huấn luyện')
    parser.add_argument('--customer-id', required=True, help='ID của khách hàng')
    parser.add_argument('--input-file', required=True, help='Đường dẫn đến file CSV dữ liệu gốc')
    parser.add_argument('--output-file', help='Đường dẫn đến file JSON kết quả (tùy chọn)')

    args = parser.parse_args()

    try:
        results = run_complete_training(
            customer_id=args.customer_id,
            input_file=args.input_file
        )

        # Lưu kết quả vào file JSON nếu được chỉ định
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu kết quả vào: {args.output_file}")

        # In ra kết quả cho script gọi
        print(json.dumps(results))

        # Trả về mã lỗi
        return 0 if results['status'] == 'success' else 1

    except Exception as e:
        logger.exception(f"Lỗi không xử lý được: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())