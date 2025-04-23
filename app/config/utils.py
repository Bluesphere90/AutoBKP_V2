import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import re
import glob
from pathlib import Path

from .paths import path_manager
from . import constants

logger = logging.getLogger(__name__)


def load_vietnamese_stopwords() -> List[str]:
    """
    Tải danh sách stopwords tiếng Việt

    Returns:
        Danh sách các stopwords tiếng Việt
    """
    stopwords_path = constants.VIETNAMESE_STOPWORDS_PATH

    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f if line.strip()]
        return stopwords
    except Exception as e:
        logger.warning(f"Không thể tải stopwords tiếng Việt từ {stopwords_path}: {str(e)}")
        return []


def generate_model_version() -> str:
    """
    Tạo phiên bản mô hình dựa trên timestamp

    Returns:
        Chuỗi phiên bản mô hình (VD: 20250423_141530)
    """
    return datetime.now().strftime(constants.MODEL_VERSION_FORMAT)


def cleanup_old_model_versions(customer_id: str, model_type: str,
                               keep_latest: int = constants.MAX_MODEL_VERSIONS) -> None:
    """
    Xóa các phiên bản mô hình cũ để giữ số lượng trong giới hạn

    Args:
        customer_id: ID của khách hàng
        model_type: Loại mô hình
        keep_latest: Số lượng phiên bản gần nhất cần giữ lại
    """
    model_dir = os.path.join(path_manager.get_customer_model_path(customer_id), model_type)

    if not os.path.exists(model_dir):
        return

    # Lấy tất cả các file mô hình
    model_pattern = os.path.join(model_dir, f"model_*.{constants.MODEL_FILE_EXT}")
    model_files = glob.glob(model_pattern)

    # Bỏ qua model_latest.joblib
    model_files = [f for f in model_files if not f.endswith(f"model_latest{constants.MODEL_FILE_EXT}")]

    if len(model_files) <= keep_latest:
        return

    # Sắp xếp theo thời gian sửa đổi (cũ nhất trước)
    model_files.sort(key=lambda x: os.path.getmtime(x))

    # Xóa các file cũ
    files_to_delete = model_files[:-keep_latest]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logger.info(f"Đã xóa phiên bản mô hình cũ: {file_path}")
        except Exception as e:
            logger.error(f"Không thể xóa file {file_path}: {str(e)}")


def is_vietnamese_text(text: str) -> bool:
    """
    Kiểm tra xem một chuỗi văn bản có phải là tiếng Việt hay không

    Args:
        text: Chuỗi văn bản cần kiểm tra

    Returns:
        True nếu văn bản chứa các ký tự tiếng Việt, False nếu không
    """
    # Các ký tự đặc trưng của tiếng Việt
    vietnamese_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"

    # Chuyển văn bản về chữ thường
    text_lower = text.lower()

    # Kiểm tra xem có ký tự tiếng Việt nào trong văn bản không
    for char in vietnamese_chars:
        if char in text_lower:
            return True

    return False


def detect_vietnamese_columns(df, sample_size: int = 100) -> List[str]:
    """
    Phát hiện tự động các cột chứa văn bản tiếng Việt

    Args:
        df: DataFrame cần kiểm tra
        sample_size: Số lượng mẫu tối đa để kiểm tra

    Returns:
        Danh sách tên các cột chứa văn bản tiếng Việt
    """
    vietnamese_columns = []

    # Chỉ kiểm tra các cột object (văn bản)
    text_columns = df.select_dtypes(include=['object']).columns

    for column in text_columns:
        # Lấy mẫu để kiểm tra
        sample = df[column].dropna().sample(min(sample_size, len(df))).astype(str).tolist()

        # Kiểm tra từng mẫu
        vietnamese_count = sum(1 for text in sample if is_vietnamese_text(text))

        # Nếu ít nhất 10% mẫu có văn bản tiếng Việt
        if vietnamese_count / len(sample) >= 0.1:
            vietnamese_columns.append(column)

    return vietnamese_columns


def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt (loại bỏ dấu câu, khoảng trắng thừa, v.v.)

    Args:
        text: Chuỗi văn bản cần chuẩn hóa

    Returns:
        Chuỗi văn bản đã được chuẩn hóa
    """
    if not isinstance(text, str):
        return ""

    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text.strip())

    # Loại bỏ dấu câu
    text = re.sub(r'[.,:#$%&()_\-"!?]', ' ', text)

    # Loại bỏ số
    text = re.sub(r'\d+', ' ', text)

    # Loại bỏ khoảng trắng thừa lần nữa
    text = re.sub(r'\s+', ' ', text.strip())

    return text.lower()


def save_metadata(path: str, metadata: Dict[str, Any]) -> None:
    """
    Lưu metadata vào file JSON

    Args:
        path: Đường dẫn đến file metadata
        metadata: Dict chứa metadata
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Đã lưu metadata tại: {path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu metadata: {str(e)}")


def load_metadata(path: str) -> Dict[str, Any]:
    """
    Tải metadata từ file JSON

    Args:
        path: Đường dẫn đến file metadata

    Returns:
        Dict chứa metadata
    """
    if not os.path.exists(path):
        logger.warning(f"Không tìm thấy file metadata: {path}")
        return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Lỗi khi tải metadata từ {path}: {str(e)}")
        return {}