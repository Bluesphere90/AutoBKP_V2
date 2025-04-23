import os
from pathlib import Path


class PathManager:
    """
    Quản lý các đường dẫn cho dự án
    """

    def __init__(self):
        # Đường dẫn gốc của dự án (thư mục cha của app)
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent

        # Đường dẫn thư mục app
        self.APP_DIR = self.PROJECT_ROOT / 'app'

        # Đường dẫn thư mục volumes (từ biến môi trường hoặc mặc định)
        self.VOLUMES_PATH = os.environ.get('VOLUMES_PATH', str(self.PROJECT_ROOT / 'volumes'))

        # Đường dẫn dữ liệu
        self.DATA_PATH = os.environ.get('DATA_PATH', str(Path(self.VOLUMES_PATH) / 'data'))
        self.RAW_DATA_PATH = os.path.join(self.DATA_PATH, 'raw')
        self.PROCESSED_DATA_PATH = os.path.join(self.DATA_PATH, 'processed')
        self.RESULTS_DATA_PATH = os.path.join(self.DATA_PATH, 'results')

        # Đường dẫn mô hình
        self.MODELS_PATH = os.environ.get('MODELS_PATH', str(Path(self.VOLUMES_PATH) / 'models'))

        # Đường dẫn cấu hình
        self.CONFIG_PATH = os.environ.get('CONFIG_PATH', str(Path(self.VOLUMES_PATH) / 'config'))
        self.APP_CONFIG_PATH = os.path.join(self.APP_DIR, 'config')

        # Đường dẫn log
        self.LOGS_PATH = os.environ.get('LOGS_PATH', str(Path(self.VOLUMES_PATH) / 'logs'))

        # Tạo các thư mục nếu chưa tồn tại
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Tạo các thư mục cần thiết nếu chưa tồn tại"""
        for path in [
            self.RAW_DATA_PATH,
            self.PROCESSED_DATA_PATH,
            self.RESULTS_DATA_PATH,
            self.MODELS_PATH,
            self.CONFIG_PATH,
            self.LOGS_PATH
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def get_customer_data_path(self, customer_id: str, data_type: str = 'raw') -> str:
        """
        Lấy đường dẫn đến dữ liệu của khách hàng

        Args:
            customer_id: ID của khách hàng
            data_type: Loại dữ liệu ('raw', 'processed', 'results')

        Returns:
            Đường dẫn đến thư mục dữ liệu của khách hàng
        """
        if data_type == 'raw':
            base_path = self.RAW_DATA_PATH
        elif data_type == 'processed':
            base_path = self.PROCESSED_DATA_PATH
        elif data_type == 'results':
            base_path = self.RESULTS_DATA_PATH
        else:
            raise ValueError(f"Loại dữ liệu không hợp lệ: {data_type}")

        customer_path = os.path.join(base_path, customer_id)
        Path(customer_path).mkdir(parents=True, exist_ok=True)

        return customer_path

    def get_customer_model_path(self, customer_id: str) -> str:
        """
        Lấy đường dẫn đến mô hình của khách hàng

        Args:
            customer_id: ID của khách hàng

        Returns:
            Đường dẫn đến thư mục mô hình của khách hàng
        """
        customer_path = os.path.join(self.MODELS_PATH, customer_id)
        Path(customer_path).mkdir(parents=True, exist_ok=True)

        return customer_path

    def get_model_file_path(self, customer_id: str, model_type: str, version: str = 'latest') -> str:
        """
        Lấy đường dẫn đến file mô hình cụ thể

        Args:
            customer_id: ID của khách hàng
            model_type: Loại mô hình ('hachtoan', 'mahanghua', 'outlier')
            version: Phiên bản mô hình ('latest' hoặc timestamp cụ thể)

        Returns:
            Đường dẫn đến file mô hình
        """
        model_dir = os.path.join(self.get_customer_model_path(customer_id), model_type)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        if version == 'latest':
            # Tìm file mô hình mới nhất
            model_files = list(Path(model_dir).glob('model_*.joblib'))
            if not model_files:
                return os.path.join(model_dir, 'model_latest.joblib')

            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            return str(latest_model)
        else:
            # Sử dụng phiên bản cụ thể
            return os.path.join(model_dir, f'model_{version}.joblib')

    def get_log_file_path(self, customer_id: str, component: str) -> str:
        """
        Lấy đường dẫn đến file log

        Args:
            customer_id: ID của khách hàng
            component: Thành phần tạo log ('training', 'prediction', 'preprocessing', etc.)

        Returns:
            Đường dẫn đến file log
        """
        customer_log_dir = os.path.join(self.LOGS_PATH, customer_id)
        Path(customer_log_dir).mkdir(parents=True, exist_ok=True)

        return os.path.join(customer_log_dir, f'{component}.log')


# Singleton instance
path_manager = PathManager()