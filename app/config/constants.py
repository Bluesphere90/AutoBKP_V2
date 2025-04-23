"""
Các hằng số và giá trị cố định được sử dụng trong dự án
"""

# Models types
MODEL_TYPES = [
    'xgboost',
    'lightgbm',
    'random_forest',
    'neural_network',
    'fasttext',
    'svm'
]

# Outlier detection methods
OUTLIER_DETECTION_METHODS = [
    'isolation_forest',
    'one_class_svm',
    'local_outlier_factor',
    'elliptic_envelope',
    'dbscan'
]

# Imbalance handling strategies
IMBALANCE_STRATEGIES = [
    'auto',                # Tự động chọn chiến lược phù hợp
    'class_weight',        # Dùng class_weight trong mô hình
    'smote',               # SMOTE oversampling
    'adasyn',              # ADASYN oversampling
    'smoteenn',            # SMOTE + ENN (over + under)
    'smotetomek',          # SMOTE + Tomek Links
    'undersampling',       # Random Undersampling
    'nearmiss',            # NearMiss Undersampling
    'ensemble',            # Ensemble (BalancedBaggingClassifier)
    'focal_loss'           # Focal Loss (cho neural networks)
]

# Vietnamese text processing libraries
VIETNAMESE_TOKENIZERS = [
    'underthesea',         # Thư viện phân tích ngôn ngữ tiếng Việt
    'pyvi',                # Thư viện xử lý tiếng Việt
    'vncorenlp',           # Vietnamese Natural Language Processing (nặng hơn)
    'spacy'                # Với mô hình tiếng Việt (nếu có)
]

# API status codes
API_STATUS = {
    'SUCCESS': 'success',
    'ERROR': 'error',
    'PROCESSING': 'processing',
    'WARNING': 'warning'
}

# Loại mô hình
MODEL_TYPE_HACHTOAN = 'hachtoan'
MODEL_TYPE_MAHANGHUA = 'mahanghua'
MODEL_TYPE_OUTLIER = 'outlier'

# Các trường mặc định trong cấu hình
DEFAULT_ID_COLUMN = 'MSTNguoiBan'
DEFAULT_TEXT_COLUMN = 'TenHangHoaDichVu'
DEFAULT_PRIMARY_TARGET = 'HachToan'
DEFAULT_SECONDARY_TARGET = 'MaHangHoa'
DEFAULT_CONDITION_PREFIX = '15'

# Tham số dataset
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# Các giá trị ngưỡng
MIN_SAMPLES_PER_CLASS = 5          # Số mẫu tối thiểu cho mỗi lớp
IMBALANCE_THRESHOLD = 10           # Ngưỡng chênh lệch tần suất để coi là mất cân bằng
SEVERE_IMBALANCE_THRESHOLD = 50    # Ngưỡng chênh lệch tần suất để coi là mất cân bằng nghiêm trọng
DEFAULT_OUTLIER_THRESHOLD = 0.85   # Ngưỡng để xác định outlier
MIN_PROBABILITY_THRESHOLD = 0.5    # Ngưỡng xác suất tối thiểu để chấp nhận dự đoán

# Thông tin mô hình
MODEL_VERSION_FORMAT = '%Y%m%d_%H%M%S'  # Định dạng phiên bản mô hình (timestamp)
MAX_MODEL_VERSIONS = 5                   # Số phiên bản tối đa lưu trữ

# File extensions
MODEL_FILE_EXT = '.joblib'
CONFIG_FILE_EXT = '.json'
DATA_FILE_EXT = '.csv'
LOG_FILE_EXT = '.log'

# API limits
MAX_BATCH_SIZE = 1000                    # Số lượng mẫu tối đa cho dự đoán hàng loạt
REQUEST_TIMEOUT = 60                      # Thời gian chờ tối đa cho request (giây)

# Training limits
MAX_TRAINING_TIME = 3600                 # Thời gian huấn luyện tối đa (giây)
EARLY_STOPPING_ROUNDS = 10               # Số vòng lặp không cải thiện để dừng sớm

# Log levels
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# Vietnamese stopwords paths
VIETNAMESE_STOPWORDS_PATH = './app/config/vietnamese_stopwords.txt'

# Thông báo cảnh báo
WARNINGS = {
    'outlier': 'Dữ liệu đầu vào có thể là outlier',
    'low_probability': 'Xác suất dự đoán thấp, độ tin cậy không cao',
    'missing_features': 'Thiếu đặc trưng đầu vào',
    'invalid_format': 'Định dạng dữ liệu không hợp lệ',
    'unknown_class': 'Lớp dự đoán không có trong dữ liệu huấn luyện'
}

# Kích thước tệp tin tối đa (bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB