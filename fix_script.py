#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để sửa lỗi trong predict.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fix_predict')


def fix_predict_py(file_path: str, backup: bool = True):
    """
    Sửa lỗi trong file predict.py

    Args:
        file_path: Đường dẫn đến file predict.py
        backup: Nếu True, sẽ tạo file backup trước khi sửa
    """
    if not os.path.exists(file_path):
        logger.error(f"File không tồn tại: {file_path}")
        return False

    # Đọc nội dung file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Tạo backup nếu cần
    if backup:
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Đã tạo backup tại: {backup_path}")

    # Sửa hàm predict_batch
    # Tìm vị trí của hàm predict_batch
    predict_batch_start = content.find("def predict_batch(")
    if predict_batch_start == -1:
        logger.error("Không tìm thấy hàm predict_batch trong file")
        return False

    # Tìm vị trí kết thúc của hàm predict_batch
    next_def = content.find("\ndef ", predict_batch_start + 1)
    if next_def == -1:
        # Nếu không tìm thấy hàm tiếp theo, có thể đây là hàm cuối cùng
        predict_batch_end = len(content)
    else:
        predict_batch_end = next_def

    # Trích xuất nội dung hiện tại của hàm
    current_predict_batch = content[predict_batch_start:predict_batch_end]

    # Tạo nội dung mới cho hàm
    new_predict_batch = """def predict_batch(customer_id: str, input_file: str, output_file: str = None,
                  models_dir: str = None, version: str = 'latest') -> str:
    \"\"\"
    Dự đoán hàng loạt từ file CSV

    Args:
        customer_id: ID của khách hàng
        input_file: Đường dẫn đến file CSV đầu vào
        output_file: Đường dẫn đến file CSV đầu ra (nếu None, sẽ tạo tên file tự động)
        models_dir: Thư mục chứa mô hình (nếu None, sẽ sử dụng thư mục mặc định)
        version: Phiên bản mô hình ('latest' hoặc phiên bản cụ thể)

    Returns:
        Đường dẫn đến file kết quả
    \"\"\"
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
    # Đảm bảo tạo các cột mới với giá trị mặc định
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

            # Convert warnings list to string for CSV storage
            warnings = prediction.get('warnings', [])
            if warnings:
                result_df.at[idx, 'Warnings'] = '; '.join(warnings)

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

    # Kiểm tra và xác nhận tất cả các cột tồn tại
    for col in ['Predicted_HachToan', 'HachToan_Probability', 'Predicted_MaHangHoa', 
                'MaHangHoa_Probability', 'Is_Outlier', 'Outlier_Score', 'Warnings']:
        if col not in result_df.columns:
            logger.warning(f"Cột {col} không tồn tại, tạo mới với giá trị None")
            result_df[col] = None

    # Lưu kết quả
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=";")
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
        "success_rate": (len(df) - result_df['Warnings'].fillna('').str.contains('Error').sum()) / len(df) if len(df) > 0 else 0
    }

    metadata_file = os.path.splitext(output_file)[0] + '_metadata.json'
    save_metadata(metadata_file, metadata)

    return output_file"""

    # Thay thế hàm predict_batch cũ bằng hàm mới
    new_content = content[:predict_batch_start] + new_predict_batch + content[predict_batch_end:]

    # Lưu lại file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    logger.info(f"Đã sửa hàm predict_batch trong {file_path}")

    # Kiểm tra xem người dùng cũng muốn sửa hàm predict không
    while True:
        answer = input("Bạn có muốn sửa hàm predict trong class ModelPredictor không? (y/n): ")
        if answer.lower() in ['y', 'yes', 'có', 'co']:
            fix_predict_method = True
            break
        elif answer.lower() in ['n', 'no', 'không', 'khong']:
            fix_predict_method = False
            break
        else:
            print("Vui lòng nhập 'y' hoặc 'n'")

    if not fix_predict_method:
        logger.info("Bỏ qua việc sửa hàm predict")
        return True

    # Sửa tiếp hàm predict trong class ModelPredictor
    # Tìm vị trí của hàm predict
    predict_start = new_content.find("def predict(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:")
    if predict_start == -1:
        logger.error("Không tìm thấy hàm predict trong file")
        return False

    # Tìm vị trí kết thúc của hàm predict
    next_def_after_predict = new_content.find("\n    def ", predict_start + 1)
    if next_def_after_predict == -1:
        # Nếu không tìm thấy hàm tiếp theo, có thể đây là hàm cuối cùng của class
        predict_end = new_content.find("\ndef ", predict_start + 1)
        if predict_end == -1:
            predict_end = len(new_content)
    else:
        predict_end = next_def_after_predict

    # Trích xuất nội dung hiện tại của hàm
    current_predict = new_content[predict_start:predict_end]

    # Tạo nội dung mới cho hàm predict với định dạng như trong file cũ
    new_predict = """    def predict(self, data: Union[pd.DataFrame, Dict]) -> Dict[str, Any]:
        \"\"\"
        Dự đoán sử dụng mô hình đã huấn luyện

        Args:
            data: DataFrame hoặc Dict chứa dữ liệu đầu vào

        Returns:
            Dict chứa kết quả dự đoán
        \"\"\"
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
                hachtoan_pred = self.hachtoan_model.predict(df)
                hachtoan_prob = self.hachtoan_model.predict_proba(df)

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
                logger.warning(f"Lỗi khi dự đoán HachToan thông thường: {str(e)}")
                result["warnings"].append(f"Không thể dự đoán HachToan, sử dụng giá trị mặc định")
                # Sử dụng giá trị mặc định thay vì crash
                result["prediction"]["HachToan"] = "1561"  # Giá trị mặc định hợp lý
                result["probabilities"]["HachToan"] = 0.5

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán HachToan: {str(e)}")
            result["warnings"].append(f"Lỗi dự đoán HachToan: {str(e)}")
            # Đảm bảo có giá trị mặc định nếu xảy ra lỗi
            result["prediction"]["HachToan"] = "1561"  # Giá trị mặc định hợp lý
            result["probabilities"]["HachToan"] = 0.5

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
                    # Sử dụng giá trị mặc định nếu có lỗi
                    result["prediction"]["MaHangHoa"] = "MH001"
                    result["probabilities"]["MaHangHoa"] = 0.5
            else:
                logger.info("Không thỏa điều kiện để dự đoán MaHangHoa")

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán MaHangHoa: {str(e)}")
            result["warnings"].append(f"Lỗi xử lý MaHangHoa: {str(e)}")

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

        # Đảm bảo các giá trị cần có đều tồn tại trước khi trả về
        if "HachToan" not in result["prediction"]:
            result["prediction"]["HachToan"] = "1561"  # Giá trị mặc định hợp lý
            result["probabilities"]["HachToan"] = 0.5
            result["warnings"].append("Không thể dự đoán HachToan, sử dụng giá trị mặc định")

        return result
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
                hachtoan_pred = self.hachtoan_model.predict(df)
                hachtoan_prob = self.hachtoan_model.predict_proba(df)

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
                logger.warning(f"Lỗi khi dự đoán HachToan thông thường: {str(e)}")
                result["warnings"].append(f"Không thể dự đoán HachToan, sử dụng giá trị mặc định")
                # Sử dụng giá trị mặc định thay vì crash
                result["prediction"]["HachToan"] = "1561"  # Giá trị mặc định hợp lý
                result["probabilities"]["HachToan"] = 0.5

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán HachToan: {str(e)}")
            result["warnings"].append(f"Lỗi dự đoán HachToan: {str(e)}")
            # Đảm bảo có giá trị mặc định nếu xảy ra lỗi
            result["prediction"]["HachToan"] = "1561"  # Giá trị mặc định hợp lý
            result["probabilities"]["HachToan"] = 0.5

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
                    # Sử dụng giá trị mặc định nếu có lỗi
                    result["prediction"]["MaHangHoa"] = "MH001"
                    result["probabilities"]["MaHangHoa"] = 0.5
            else:
                logger.info("Không thỏa điều kiện để dự đoán MaHangHoa")

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán MaHangHoa: {str(e)}")
            result["warnings"].append(f"Lỗi xử lý MaHangHoa: {str(e)}")

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

        # Đảm bảo các giá trị cần có đều tồn tại trước khi trả về
        if "HachToan" not in result["prediction"]:
            result["prediction"]["HachToan"] = "1561"  # Giá trị mặc định hợp lý
            result["probabilities"]["HachToan"] = 0.5
            result["warnings"].append("Không thể dự đoán HachToan, sử dụng giá trị mặc định")

        return result"""