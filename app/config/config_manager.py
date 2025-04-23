import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import copy
import datetime

# Thiết lập logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('config_manager')


class ConfigManager:
    """
    Lớp quản lý cấu hình cho dự án machine learning
    - Tải cấu hình mặc định
    - Quản lý cấu hình riêng cho từng khách hàng
    - Lưu và cập nhật cấu hình
    """

    def __init__(self, config_path: str = None, volumes_path: str = None):
        """
        Khởi tạo ConfigManager

        Args:
            config_path: Đường dẫn đến thư mục cấu hình ứng dụng
            volumes_path: Đường dẫn đến thư mục volumes
        """
        # Đường dẫn cấu hình
        self.app_config_path = config_path or os.environ.get('APP_CONFIG_PATH', './app/config')
        self.volumes_path = volumes_path or os.environ.get('VOLUMES_PATH', './volumes')
        self.volumes_config_path = os.path.join(self.volumes_path, 'config')

        # Đảm bảo thư mục tồn tại
        Path(self.volumes_config_path).mkdir(parents=True, exist_ok=True)

        # Tải cấu hình mặc định
        self.default_config = self._load_default_config()
        logger.info("Đã tải cấu hình mặc định")

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Tải cấu hình mặc định từ file

        Returns:
            Dict chứa cấu hình mặc định
        """
        default_config_path = os.path.join(self.app_config_path, 'default_config.json')

        try:
            with open(default_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file cấu hình mặc định tại {default_config_path}")
            # Trả về cấu hình rỗng nếu không tìm thấy file
            return {}
        except json.JSONDecodeError:
            logger.error(f"Lỗi khi phân tích file cấu hình mặc định {default_config_path}")
            return {}

    def get_customer_config(self, customer_id: str) -> Dict[str, Any]:
        """
        Lấy cấu hình cho một khách hàng cụ thể
        Nếu chưa có, sẽ tạo cấu hình mới từ mẫu mặc định

        Args:
            customer_id: ID của khách hàng

        Returns:
            Dict chứa cấu hình của khách hàng
        """
        config_path = os.path.join(self.volumes_config_path, f"{customer_id}.json")

        # Nếu file cấu hình tồn tại, tải nó
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Lỗi khi tải cấu hình khách hàng {customer_id}: {str(e)}")
                # Nếu có lỗi, trả về cấu hình mặc định
                return copy.deepcopy(self.default_config)

        # Nếu không tìm thấy, tạo từ cấu hình mặc định
        logger.info(f"Tạo cấu hình mới cho khách hàng {customer_id}")
        customer_config = copy.deepcopy(self.default_config)
        self.save_customer_config(customer_id, customer_config)

        return customer_config

    def update_customer_config(self, customer_id: str, new_config: Dict[str, Any],
                               partial_update: bool = True) -> Dict[str, Any]:
        """
        Cập nhật cấu hình cho khách hàng

        Args:
            customer_id: ID của khách hàng
            new_config: Cấu hình mới (có thể là một phần)
            partial_update: Nếu True, chỉ cập nhật các trường được cung cấp
                           Nếu False, thay thế toàn bộ cấu hình cũ

        Returns:
            Dict chứa cấu hình đã cập nhật
        """
        if partial_update:
            # Tải cấu hình hiện tại
            current_config = self.get_customer_config(customer_id)

            # Cập nhật đệ quy
            updated_config = self._recursive_update(current_config, new_config)
        else:
            # Thay thế hoàn toàn
            updated_config = copy.deepcopy(new_config)

        # Lưu cấu hình cập nhật
        self.save_customer_config(customer_id, updated_config)

        return updated_config

    def _recursive_update(self, base_config: Dict[str, Any],
                          update_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cập nhật đệ quy các trường trong dict

        Args:
            base_config: Dict cơ sở
            update_config: Dict chứa các cập nhật

        Returns:
            Dict đã được cập nhật
        """
        result = copy.deepcopy(base_config)

        for key, value in update_config.items():
            # Nếu cả hai đều là dict, cập nhật đệ quy
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._recursive_update(result[key], value)
            # Nếu không, thay thế hoặc thêm mới
            else:
                result[key] = copy.deepcopy(value)

        return result

    def save_customer_config(self, customer_id: str, config: Dict[str, Any]) -> str:
        """
        Lưu cấu hình của khách hàng

        Args:
            customer_id: ID của khách hàng
            config: Cấu hình cần lưu

        Returns:
            Đường dẫn đến file cấu hình đã lưu
        """
        # Thêm metadata
        config['_metadata'] = {
            'last_updated': datetime.datetime.now().isoformat(),
            'customer_id': customer_id
        }

        config_path = os.path.join(self.volumes_config_path, f"{customer_id}.json")

        # Lưu backup nếu file đã tồn tại
        if os.path.exists(config_path):
            backup_dir = os.path.join(self.volumes_config_path, 'backups', customer_id)
            Path(backup_dir).mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f"{customer_id}_{timestamp}.json")

            try:
                with open(config_path, 'r', encoding='utf-8') as src, \
                        open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
                logger.info(f"Đã tạo backup cấu hình tại {backup_path}")
            except Exception as e:
                logger.warning(f"Không thể tạo backup cấu hình: {str(e)}")

        # Lưu cấu hình mới
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu cấu hình cho khách hàng {customer_id}")
            return config_path
        except Exception as e:
            logger.error(f"Lỗi khi lưu cấu hình khách hàng {customer_id}: {str(e)}")
            raise

    def list_customers(self) -> List[str]:
        """
        Liệt kê danh sách khách hàng có cấu hình

        Returns:
            Danh sách ID của các khách hàng
        """
        configs = list(Path(self.volumes_config_path).glob('*.json'))
        return [config.stem for config in configs]

    def delete_customer_config(self, customer_id: str) -> bool:
        """
        Xóa cấu hình của khách hàng

        Args:
            customer_id: ID của khách hàng

        Returns:
            True nếu xóa thành công, False nếu không
        """
        config_path = os.path.join(self.volumes_config_path, f"{customer_id}.json")

        if not os.path.exists(config_path):
            logger.warning(f"Không tìm thấy cấu hình cho khách hàng {customer_id}")
            return False

        # Tạo backup trước khi xóa
        backup_dir = os.path.join(self.volumes_config_path, 'backups', customer_id)
        Path(backup_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(backup_dir, f"{customer_id}_{timestamp}_deleted.json")

        try:
            with open(config_path, 'r', encoding='utf-8') as src, \
                    open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            logger.info(f"Đã tạo backup cấu hình trước khi xóa tại {backup_path}")

            # Xóa file cấu hình
            os.remove(config_path)
            logger.info(f"Đã xóa cấu hình khách hàng {customer_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa cấu hình khách hàng {customer_id}: {str(e)}")
            return False

    def reset_customer_config(self, customer_id: str) -> Dict[str, Any]:
        """
        Đặt lại cấu hình của khách hàng về mặc định

        Args:
            customer_id: ID của khách hàng

        Returns:
            Dict chứa cấu hình mặc định
        """
        default_config = copy.deepcopy(self.default_config)
        self.save_customer_config(customer_id, default_config)
        logger.info(f"Đã đặt lại cấu hình của khách hàng {customer_id} về mặc định")
        return default_config

    def get_model_config(self, customer_id: str, model_type: str) -> Dict[str, Any]:
        """
        Lấy cấu hình mô hình cụ thể cho khách hàng

        Args:
            customer_id: ID của khách hàng
            model_type: Loại mô hình ('hachtoan_model', 'mahanghua_model', 'outlier_detection')

        Returns:
            Dict chứa cấu hình mô hình
        """
        customer_config = self.get_customer_config(customer_id)

        if 'model_config' not in customer_config or model_type not in customer_config['model_config']:
            logger.warning(f"Không tìm thấy cấu hình cho mô hình {model_type} của khách hàng {customer_id}")
            # Trả về cấu hình mặc định cho loại mô hình này
            if model_type in self.default_config.get('model_config', {}):
                return copy.deepcopy(self.default_config['model_config'][model_type])
            return {}

        return copy.deepcopy(customer_config['model_config'][model_type])

    def get_training_config(self, customer_id: str) -> Dict[str, Any]:
        """
        Lấy cấu hình huấn luyện cho khách hàng

        Args:
            customer_id: ID của khách hàng

        Returns:
            Dict chứa cấu hình huấn luyện
        """
        customer_config = self.get_customer_config(customer_id)

        if 'training_config' not in customer_config:
            logger.warning(f"Không tìm thấy cấu hình huấn luyện cho khách hàng {customer_id}")
            # Trả về cấu hình huấn luyện mặc định
            if 'training_config' in self.default_config:
                return copy.deepcopy(self.default_config['training_config'])
            return {}

        return copy.deepcopy(customer_config['training_config'])

    def get_preprocessing_config(self, customer_id: str) -> Dict[str, Any]:
        """
        Lấy cấu hình tiền xử lý cho khách hàng

        Args:
            customer_id: ID của khách hàng

        Returns:
            Dict chứa cấu hình tiền xử lý
        """
        customer_config = self.get_customer_config(customer_id)

        if 'preprocessing_config' not in customer_config:
            logger.warning(f"Không tìm thấy cấu hình tiền xử lý cho khách hàng {customer_id}")
            # Trả về cấu hình tiền xử lý mặc định
            if 'preprocessing_config' in self.default_config:
                return copy.deepcopy(self.default_config['preprocessing_config'])
            return {}

        return copy.deepcopy(customer_config['preprocessing_config'])

    def get_column_config(self, customer_id: str) -> Dict[str, Any]:
        """
        Lấy cấu hình cột cho khách hàng

        Args:
            customer_id: ID của khách hàng

        Returns:
            Dict chứa cấu hình cột
        """
        customer_config = self.get_customer_config(customer_id)

        if 'column_config' not in customer_config:
            logger.warning(f"Không tìm thấy cấu hình cột cho khách hàng {customer_id}")
            # Trả về cấu hình cột mặc định
            if 'column_config' in self.default_config:
                return copy.deepcopy(self.default_config['column_config'])
            return {}

        return copy.deepcopy(customer_config['column_config'])


# Singleton instance
config_manager = ConfigManager()