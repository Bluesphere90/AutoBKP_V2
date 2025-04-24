#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script kiểm tra ứng dụng AutoBKP
- Kiểm tra cấu hình
- Kiểm tra tiền xử lý dữ liệu
- Kiểm tra huấn luyện mô hình
- Kiểm tra dự đoán
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import time
import csv
import random
from datetime import datetime
from typing import Dict, List, Any, Union

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='test_autobkp.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('test_autobkp')

# Import các module cần thiết
try:
    from app.config import config_manager, path_manager
    from app.scripts.validate_data import validate_data
    from app.scripts.preprocess import prepare_data_for_training
    from app.scripts.train import train_customer_model
    from app.scripts.predict import predict_single_sample, predict_batch

    CONFIG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.error(f"Lỗi khi import các module: {str(e)}")
    CONFIG_IMPORT_SUCCESS = False


class AutoBKPTester:
    """
    Lớp kiểm tra ứng dụng AutoBKP
    """

    def __init__(self, customer_id: str = "test_customer", temp_dir: str = None):
        """
        Khởi tạo AutoBKPTester

        Args:
            customer_id: ID của khách hàng test
            temp_dir: Thư mục tạm để lưu các file test
        """
        self.customer_id = customer_id

        # Thiết lập thư mục tạm
        if temp_dir:
            self.temp_dir = temp_dir
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="autobkp_test_")

        # Đường dẫn đến file dữ liệu giả
        self.sample_data_path = os.path.join(self.temp_dir, f"{customer_id}_sample_data.csv")

        # Đường dẫn các file kết quả
        self.train_file = None
        self.test_file = None
        self.model_files = {}
        self.prediction_file = None

        # Trạng thái các bài test
        self.test_results = {
            "config": False,
            "sample_data": False,
            "validate": False,
            "preprocess": False,
            "train": False,
            "predict": False
        }

        logger.info(f"Đã khởi tạo AutoBKPTester với customer_id: {customer_id}")
        logger.info(f"Thư mục tạm: {self.temp_dir}")

    def cleanup(self):
        """Dọn dẹp các file tạm"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Đã xóa thư mục tạm: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp thư mục tạm: {str(e)}")

    def test_config(self) -> bool:
        """
        Kiểm tra cấu hình

        Returns:
            True nếu kiểm tra thành công, False nếu thất bại
        """
        logger.info("Bắt đầu kiểm tra cấu hình...")

        try:
            if not CONFIG_IMPORT_SUCCESS:
                logger.error("Không thể import các module cần thiết")
                return False

            # Kiểm tra cấu hình mặc định
            default_config = config_manager.default_config
            if not default_config:
                logger.error("Không thể tải cấu hình mặc định")
                return False

            logger.info("Đã tải cấu hình mặc định thành công")

            # Kiểm tra cấu hình khách hàng
            customer_config = config_manager.get_customer_config(self.customer_id)
            if not customer_config:
                logger.error(f"Không thể tải cấu hình khách hàng {self.customer_id}")
                return False

            logger.info(f"Đã tải cấu hình khách hàng {self.customer_id} thành công")

            # Kiểm tra các thành phần quan trọng của cấu hình
            required_configs = [
                "column_config",
                "preprocessing_config",
                "model_config",
                "training_config"
            ]

            for config_name in required_configs:
                if config_name not in customer_config:
                    logger.error(f"Thiếu thành phần cấu hình: {config_name}")
                    return False

            # Kiểm tra cấu hình cột
            column_config = customer_config.get("column_config", {})
            if not column_config.get("vietnamese_text_columns"):
                logger.error("Thiếu cấu hình vietnamese_text_columns")
                return False

            if not column_config.get("id_columns"):
                logger.error("Thiếu cấu hình id_columns")
                return False

            if not column_config.get("target_columns", {}).get("primary"):
                logger.error("Thiếu cấu hình target_columns.primary")
                return False

            # Kiểm tra thư mục volumes
            volumes_path = path_manager.VOLUMES_PATH
            if not os.path.exists(volumes_path):
                os.makedirs(volumes_path, exist_ok=True)
                logger.info(f"Đã tạo thư mục volumes: {volumes_path}")

            logger.info("Kiểm tra cấu hình thành công")
            self.test_results["config"] = True
            return True

        except Exception as e:
            logger.exception(f"Lỗi khi kiểm tra cấu hình: {str(e)}")
            return False

    def generate_sample_data(self, num_rows: int = 100) -> bool:
        """
        Tạo dữ liệu mẫu để test

        Args:
            num_rows: Số dòng dữ liệu mẫu

        Returns:
            True nếu tạo thành công, False nếu thất bại
        """
        logger.info(f"Bắt đầu tạo {num_rows} dòng dữ liệu mẫu...")

        try:
            # Tạo danh sách các giá trị mẫu
            businesses = ["Công ty TNHH", "Công ty Cổ phần", "Doanh nghiệp Tư nhân", "Hộ kinh doanh"]
            business_fields = ["Thương mại", "Dịch vụ", "Sản xuất", "Xây dựng", "Vận tải"]
            product_types = ["Văn phòng phẩm", "Thiết bị điện tử", "Vật liệu xây dựng", "Thực phẩm", "Dịch vụ tư vấn"]

            product_names = [
                "Máy tính xách tay", "Điện thoại di động", "Bàn ghế văn phòng", "Giấy in A4", "Mực in",
                "Xi măng", "Sắt thép xây dựng", "Gạch ốp lát", "Sơn nước", "Ống nước PVC",
                "Dịch vụ kế toán", "Dịch vụ bảo vệ", "Dịch vụ vận chuyển", "Dịch vụ tư vấn", "Dịch vụ thiết kế"
            ]

            # Các mã hạch toán mẫu
            hachtoan_codes = [
                "1561", "1562", "1563", "1564", "1565",  # Bắt đầu bằng 15 cho MaHangHoa
                "2111", "2112", "2113", "2114", "2115",
                "3331", "3332", "3333", "3334", "3335"
            ]

            # Các mã hàng hóa mẫu (chỉ cho các mã hạch toán bắt đầu bằng 15)
            mahanghua_codes = [
                "MH001", "MH002", "MH003", "MH004", "MH005",
                "MH006", "MH007", "MH008", "MH009", "MH010"
            ]

            # Tạo dữ liệu ngẫu nhiên
            data = []
            for i in range(num_rows):
                mst = f"{random.randint(1000000000, 9999999999)}"
                business_name = f"{random.choice(businesses)} {random.choice(business_fields)} {mst[:4]}"

                product_name = random.choice(product_names)
                full_product_name = f"{product_name} {random.choice(product_types)}"

                hachtoan = random.choice(hachtoan_codes)

                # Chỉ gán MaHangHoa nếu HachToan bắt đầu bằng "15"
                if hachtoan.startswith("15"):
                    mahanghua = random.choice(mahanghua_codes)
                else:
                    mahanghua = ""

                row = {
                    "MSTNguoiBan": mst,
                    "TenNguoiBan": business_name,
                    "TenHangHoaDichVu": full_product_name,
                    "SoLuong": random.randint(1, 100),
                    "DonGia": random.randint(10000, 10000000),
                    "HachToan": hachtoan,
                    "MaHangHoa": mahanghua
                }

                data.append(row)

            # Lưu vào file CSV
            with open(self.sample_data_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0].keys()), delimiter=';')
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Đã tạo dữ liệu mẫu và lưu vào: {self.sample_data_path}")
            self.test_results["sample_data"] = True
            return True

        except Exception as e:
            logger.exception(f"Lỗi khi tạo dữ liệu mẫu: {str(e)}")
            return False

    def test_validate_data(self) -> bool:
        """
        Kiểm tra chức năng validate_data

        Returns:
            True nếu kiểm tra thành công, False nếu thất bại
        """
        logger.info("Bắt đầu kiểm tra chức năng validate_data...")

        try:
            if not self.test_results["sample_data"]:
                logger.error("Chưa tạo dữ liệu mẫu")
                return False

            # Gọi hàm validate_data
            result = validate_data(self.sample_data_path, self.customer_id)

            # Kiểm tra kết quả
            if result["status"] != "success":
                logger.error(f"Lỗi khi validate dữ liệu: {result.get('error', 'Unknown error')}")
                return False

            logger.info(f"Kết quả validate: {result['overall_status']}")

            # Ghi kết quả chi tiết vào file
            result_path = os.path.join(self.temp_dir, "validate_result.json")

            # Hàm xử lý các kiểu dữ liệu đặc biệt cho JSON
            def json_serializable(obj):
                # Xử lý các kiểu numpy
                if isinstance(obj, np.generic):
                    return obj.item()  # Chuyển tất cả các kiểu numpy thành kiểu Python tương ứng
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # Chuyển numpy array thành list
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict()  # Chuyển DataFrame thành dict
                if isinstance(obj, pd.Series):
                    return obj.to_dict()  # Chuyển Series thành dict
                # Nếu không thuộc các kiểu trên, báo lỗi
                raise TypeError(f"Type {type(obj)} not serializable")

            # Lưu kết quả với custom encoder
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=json_serializable)

            logger.info(f"Đã lưu kết quả validate vào: {result_path}")

            # Kiểm tra các warnings
            if len(result.get("warnings", [])) > 0:
                logger.warning(f"Tìm thấy {len(result['warnings'])} cảnh báo trong dữ liệu")
                for warning in result["warnings"]:
                    logger.warning(f"- {warning['message']} (severity: {warning['severity']})")

            logger.info("Kiểm tra chức năng validate_data thành công")
            self.test_results["validate"] = True
            return True

        except Exception as e:
            logger.exception(f"Lỗi khi kiểm tra chức năng validate_data: {str(e)}")
            return False

    def test_preprocess(self) -> bool:
        """
        Kiểm tra chức năng tiền xử lý dữ liệu

        Returns:
            True nếu kiểm tra thành công, False nếu thất bại
        """
        logger.info("Bắt đầu kiểm tra chức năng tiền xử lý dữ liệu...")

        try:
            if not self.test_results["sample_data"]:
                logger.error("Chưa tạo dữ liệu mẫu")
                return False

            # Gọi hàm prepare_data_for_training
            result = prepare_data_for_training(
                customer_id=self.customer_id,
                input_file=self.sample_data_path,
                output_dir=self.temp_dir
            )

            # Kiểm tra kết quả
            if not result or not result.get("train_file") or not result.get("test_file"):
                logger.error("Lỗi khi tiền xử lý dữ liệu: không nhận được kết quả hợp lệ")
                return False

            self.train_file = result["train_file"]
            self.test_file = result["test_file"]

            logger.info(f"Đã tiền xử lý dữ liệu thành công")
            logger.info(f"- Tập huấn luyện: {self.train_file}")
            logger.info(f"- Tập kiểm tra: {self.test_file}")

            # Kiểm tra file đã tạo
            if not os.path.exists(self.train_file):
                logger.error(f"File tập huấn luyện không tồn tại: {self.train_file}")
                return False

            if not os.path.exists(self.test_file):
                logger.error(f"File tập kiểm tra không tồn tại: {self.test_file}")
                return False

            # Đọc dữ liệu đã xử lý
            train_df = pd.read_csv(self.train_file, sep=";", encoding='utf-8-sig')
            test_df = pd.read_csv(self.test_file, sep=";", encoding='utf-8-sig')

            logger.info(f"Số lượng mẫu trong tập huấn luyện: {len(train_df)}")
            logger.info(f"Số lượng mẫu trong tập kiểm tra: {len(test_df)}")

            # Kiểm tra các cột quan trọng
            column_config = config_manager.get_column_config(self.customer_id)
            primary_target = column_config.get('target_columns', {}).get('primary')

            if primary_target not in train_df.columns:
                logger.error(f"Thiếu cột mục tiêu chính {primary_target} trong tập huấn luyện")
                return False

            logger.info("Kiểm tra chức năng tiền xử lý dữ liệu thành công")
            self.test_results["preprocess"] = True
            return True

        except Exception as e:
            logger.exception(f"Lỗi khi kiểm tra chức năng tiền xử lý dữ liệu: {str(e)}")
            return False

    def test_train(self) -> bool:
        """
        Kiểm tra chức năng huấn luyện mô hình

        Returns:
            True nếu kiểm tra thành công, False nếu thất bại
        """
        logger.info("Bắt đầu kiểm tra chức năng huấn luyện mô hình...")

        try:
            if not self.test_results["preprocess"]:
                logger.error("Chưa tiền xử lý dữ liệu")
                return False

            # Gọi hàm train_customer_model
            result = train_customer_model(
                customer_id=self.customer_id,
                train_file=self.train_file,
                test_file=self.test_file
            )

            # Kiểm tra kết quả
            if result["status"] != "success":
                logger.error(f"Lỗi khi huấn luyện mô hình: {result.get('error', 'Unknown error')}")
                return False

            logger.info(f"Đã huấn luyện mô hình thành công trong {result.get('total_time', 0):.2f} giây")

            # Lưu các đường dẫn đến file mô hình
            self.model_files = result.get("saved_files", {})

            # Kiểm tra các file mô hình đã tạo
            if "hachtoan_model" in self.model_files:
                logger.info(f"Mô hình HachToan: {self.model_files['hachtoan_model']}")
                if not os.path.exists(self.model_files["hachtoan_model"]):
                    logger.error(f"File mô hình HachToan không tồn tại")
                    return False
            else:
                logger.warning("Không tìm thấy thông tin về mô hình HachToan")

            if "mahanghua_model" in self.model_files:
                logger.info(f"Mô hình MaHangHoa: {self.model_files['mahanghua_model']}")
                if not os.path.exists(self.model_files["mahanghua_model"]):
                    logger.error(f"File mô hình MaHangHoa không tồn tại")
                    return False
            else:
                logger.warning("Không tìm thấy thông tin về mô hình MaHangHoa")

            logger.info("Kiểm tra chức năng huấn luyện mô hình thành công")
            self.test_results["train"] = True
            return True

        except Exception as e:
            logger.exception(f"Lỗi khi kiểm tra chức năng huấn luyện mô hình: {str(e)}")
            return False

    def test_predict(self) -> bool:
        """
        Kiểm tra chức năng dự đoán

        Returns:
            True nếu kiểm tra thành công, False nếu thất bại
        """
        logger.info("Bắt đầu kiểm tra chức năng dự đoán...")

        try:
            if not self.test_results["train"]:
                logger.error("Chưa huấn luyện mô hình")
                return False

            # Kiểm tra dự đoán đơn lẻ
            test_data = {
                "MSTNguoiBan": "1234567890",
                "TenHangHoaDichVu": "Máy tính xách tay Dell Inspiron"
            }

            # Gọi hàm predict_single_sample
            single_result = predict_single_sample(self.customer_id, test_data)

            logger.info("Kết quả dự đoán đơn lẻ:")
            logger.info(f"- HachToan: {single_result.get('prediction', {}).get('HachToan')}")
            logger.info(f"- MaHangHoa: {single_result.get('prediction', {}).get('MaHangHoa')}")
            logger.info(f"- Outlier: {single_result.get('outlier_warning', False)}")

            # Kiểm tra dự đoán hàng loạt
            batch_result_file = predict_batch(
                customer_id=self.customer_id,
                input_file=self.sample_data_path,
                output_file=os.path.join(self.temp_dir, f"{self.customer_id}_predictions.csv")
            )

            self.prediction_file = batch_result_file

            logger.info(f"Đã dự đoán hàng loạt, kết quả lưu tại: {self.prediction_file}")

            # Kiểm tra file kết quả dự đoán
            if not os.path.exists(self.prediction_file):
                logger.error(f"File kết quả dự đoán không tồn tại")
                return False

            # Đọc kết quả dự đoán
            pred_df = pd.read_csv(self.prediction_file, sep=";", encoding='utf-8-sig')

            logger.info(f"Số lượng mẫu đã dự đoán: {len(pred_df)}")
            logger.info(f"Số lượng dự đoán HachToan: {pred_df['Predicted_HachToan'].notna().sum()}")
            logger.info(f"Số lượng dự đoán MaHangHoa: {pred_df['Predicted_MaHangHoa'].notna().sum()}")
            logger.info(f"Số lượng outlier: {pred_df['Is_Outlier'].sum()}")

            logger.info("Kiểm tra chức năng dự đoán thành công")
            self.test_results["predict"] = True
            return True

        except Exception as e:
            logger.exception(f"Lỗi khi kiểm tra chức năng dự đoán: {str(e)}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """
        Chạy tất cả các bài kiểm tra

        Returns:
            Dict chứa kết quả của các bài kiểm tra
        """
        logger.info("=== BẮT ĐẦU CHẠY TẤT CẢ CÁC BÀI KIỂM TRA ===")

        # Kiểm tra cấu hình
        if self.test_config():
            logger.info("✓ Kiểm tra cấu hình thành công")
        else:
            logger.error("✗ Kiểm tra cấu hình thất bại")
            logger.warning("Dừng các bài kiểm tra tiếp theo")
            return self.test_results

        # Tạo dữ liệu mẫu
        if self.generate_sample_data():
            logger.info("✓ Tạo dữ liệu mẫu thành công")
        else:
            logger.error("✗ Tạo dữ liệu mẫu thất bại")
            logger.warning("Dừng các bài kiểm tra tiếp theo")
            return self.test_results

        # Kiểm tra validate_data
        if self.test_validate_data():
            logger.info("✓ Kiểm tra chức năng validate_data thành công")
        else:
            logger.error("✗ Kiểm tra chức năng validate_data thất bại")
            logger.warning("Dừng các bài kiểm tra tiếp theo")
            return self.test_results

        # Kiểm tra tiền xử lý
        if self.test_preprocess():
            logger.info("✓ Kiểm tra chức năng tiền xử lý thành công")
        else:
            logger.error("✗ Kiểm tra chức năng tiền xử lý thất bại")
            logger.warning("Dừng các bài kiểm tra tiếp theo")
            return self.test_results

        # Kiểm tra huấn luyện
        if self.test_train():
            logger.info("✓ Kiểm tra chức năng huấn luyện thành công")
        else:
            logger.error("✗ Kiểm tra chức năng huấn luyện thất bại")
            logger.warning("Dừng các bài kiểm tra tiếp theo")
            return self.test_results

        # Kiểm tra dự đoán
        if self.test_predict():
            logger.info("✓ Kiểm tra chức năng dự đoán thành công")
        else:
            logger.error("✗ Kiểm tra chức năng dự đoán thất bại")

        logger.info("=== KẾT THÚC CÁC BÀI KIỂM TRA ===")

        # Tóm tắt kết quả
        success_count = sum(1 for result in self.test_results.values() if result)
        total_count = len(self.test_results)

        logger.info(f"Tổng kết: {success_count}/{total_count} bài kiểm tra thành công")

        return self.test_results


def main():
    """Hàm chính để chạy script từ command line"""
    parser = argparse.ArgumentParser(description='Kiểm tra ứng dụng AutoBKP')
    parser.add_argument('--customer-id', default='test_customer', help='ID của khách hàng test')
    parser.add_argument('--temp-dir', help='Thư mục tạm để lưu các file test (tùy chọn)')
    parser.add_argument('--cleanup', action='store_true', help='Xóa các file tạm sau khi test')

    args = parser.parse_args()

    try:
        # Khởi tạo tester
        tester = AutoBKPTester(customer_id=args.customer_id, temp_dir=args.temp_dir)

        # Chạy tất cả các bài kiểm tra
        results = tester.run_all_tests()

        # Hiển thị tóm tắt
        print("\n=== KẾT QUẢ KIỂM TRA ===")

        all_success = True
        for test_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {test_name}")
            if not success:
                all_success = False

        # Dọn dẹp nếu được yêu cầu
        if args.cleanup:
            tester.cleanup()
        else:
            print(f"\nCác file test được lưu tại: {tester.temp_dir}")

        return 0 if all_success else 1

    except Exception as e:
        logger.exception(f"Lỗi không xử lý được: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())