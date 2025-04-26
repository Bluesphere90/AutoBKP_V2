"""
Các công cụ trích xuất đặc trưng cho dữ liệu
- Xử lý văn bản tiếng Việt
- Tạo các đặc trưng từ dữ liệu cấu trúc
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
import logging

# Thử import các thư viện xử lý tiếng Việt
try:
    import underthesea
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False

try:
    import pyvi
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Định nghĩa hàm global để có thể pickle
def preprocess_vietnamese_text_global(text):
    """
    Hàm tiền xử lý văn bản tiếng Việt ở mức global
    """
    # Xử lý giá trị đơn
    if not isinstance(text, str):
        return "" if pd.isna(text) else str(text)

    # Chuyển về chữ thường
    text = text.lower()

    # Loại bỏ dấu câu và ký tự đặc biệt
    text = re.sub(r'[^\w\s]', ' ', text)

    # Loại bỏ số => giữ lại số
    # text = re.sub(r'\d+', ' ', text)

    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # Tách từ tiếng Việt nếu có thư viện hỗ trợ
    if UNDERTHESEA_AVAILABLE:
        try:
            # Sử dụng underthesea để tách từ
            tokens = underthesea.word_tokenize(text)
            text = ' '.join(tokens)
        except Exception as e:
            logger.warning(f"Lỗi khi tách từ với underthesea: {str(e)}")
    elif PYVI_AVAILABLE:
        try:
            # Sử dụng pyvi để tách từ
            text = ViTokenizer.tokenize(text)
        except Exception as e:
            logger.warning(f"Lỗi khi tách từ với pyvi: {str(e)}")

    return text

# Định nghĩa các hàm xử lý series/dataframe ở mức global
def preprocess_text_series_global(series):
    """
    Xử lý một pandas Series với văn bản tiếng Việt
    """
    return series.apply(preprocess_vietnamese_text_global)

def preprocess_text_array_global(array):
    """
    Xử lý một numpy array với văn bản tiếng Việt
    """
    return np.array([preprocess_vietnamese_text_global(text) for text in array])


class PicklableColumnPreprocessor:
    """
    Lớp preprocessor cho cột có thể pickle
    """
    def __init__(self, column):
        self.column = column

    def __call__(self, X):
        if isinstance(X, pd.DataFrame):
            return preprocess_text_series_global(X[self.column])
        elif isinstance(X, pd.Series):
            return preprocess_text_series_global(X)
        elif isinstance(X, np.ndarray):
            if X.ndim > 1:
                # Nếu là ma trận 2D, giả sử cột đầu tiên chứa văn bản
                return preprocess_text_array_global(X[:, 0])
            else:
                return preprocess_text_array_global(X)
        else:
            return preprocess_vietnamese_text_global(X)

    def __reduce__(self):
        """
        Hỗ trợ pickling bằng cách xác định cách khôi phục đối tượng
        """
        return (self.__class__, (self.column,))


class FeatureExtractor:
    """
    Lớp trích xuất đặc trưng từ dữ liệu thô
    - Hỗ trợ xử lý văn bản tiếng Việt
    - Hỗ trợ mã hóa biến phân loại
    - Tạo đặc trưng tùy chỉnh
    """

    def __init__(self, text_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 numeric_columns: List[str] = None,
                 config: Dict[str, Any] = None):
        """
        Khởi tạo FeatureExtractor

        Args:
            text_columns: Danh sách cột văn bản
            categorical_columns: Danh sách cột phân loại
            numeric_columns: Danh sách cột số
            config: Cấu hình trích xuất đặc trưng
        """
        self.text_columns = text_columns or []
        self.categorical_columns = categorical_columns or []
        self.numeric_columns = numeric_columns or []
        self.config = config or {}

        self.text_transformers = {}
        self.categorical_encoders = {}
        self.label_encoders = {}
        self.column_transformer = None
        self.feature_names = None

        # Kiểm tra thư viện xử lý tiếng Việt
        if self.text_columns and not (UNDERTHESEA_AVAILABLE or PYVI_AVAILABLE):
            logger.warning(
                "Không tìm thấy thư viện xử lý tiếng Việt (underthesea, pyvi). "
                "Chức năng xử lý tiếng Việt sẽ bị hạn chế."
            )

    def _create_text_transformer(self, column: str) -> Pipeline:
        """
        Tạo transformer cho cột văn bản

        Args:
            column: Tên cột văn bản

        Returns:
            Pipeline xử lý văn bản
        """
        # Lấy cấu hình
        text_config = self.config.get('text_features', {})
        max_features = text_config.get('max_features', 10000)
        ngram_range = tuple(text_config.get('ngram_range', [1, 2]))
        min_df = text_config.get('min_df', 2)

        # Tạo vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            lowercase=True,
            strip_accents='unicode'
        )

        # Lưu vectorizer
        self.text_transformers[column] = vectorizer

        # Tạo một đối tượng có thể pickle cho tiền xử lý
        preprocessor = PicklableColumnPreprocessor(column)

        # Sử dụng FunctionTransformer từ scikit-learn
        preprocess_transformer = FunctionTransformer(
            preprocessor,
            validate=False
        )

        return Pipeline([
            ('preprocess', preprocess_transformer),
            ('vectorize', vectorizer)
        ])

    def fit(self, df: pd.DataFrame, y: pd.Series = None) -> 'FeatureExtractor':
        """
        Học các transformer từ dữ liệu

        Args:
            df: DataFrame chứa dữ liệu
            y: Series chứa biến mục tiêu (tùy chọn)

        Returns:
            FeatureExtractor đã được fit
        """
        # Tạo danh sách transformer
        transformers = []

        # Xử lý cột văn bản
        for col in self.text_columns:
            if col in df.columns:
                transformers.append(
                    (f'text_{col}', self._create_text_transformer(col), col)
                )

        # Xử lý cột phân loại
        for col in self.categorical_columns:
            if col in df.columns:
                # Thay đổi từ sparse=False sang sparse_output=False
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self.categorical_encoders[col] = encoder
                transformers.append((f'cat_{col}', encoder, [col]))

        # Xử lý cột số
        numeric_transformer = StandardScaler()
        if self.numeric_columns:
            numeric_cols = [col for col in self.numeric_columns if col in df.columns]
            if numeric_cols:
                transformers.append(('num', numeric_transformer, numeric_cols))

        # Tạo column transformer
        self.column_transformer = ColumnTransformer(transformers, force_int_remainder_cols=False)

        # Fit transformer
        self.column_transformer.fit(df)

        # Lưu tên các đặc trưng
        self.feature_names = self._get_feature_names()

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Chuyển đổi dữ liệu thành ma trận đặc trưng

        Args:
            df: DataFrame chứa dữ liệu

        Returns:
            Ma trận đặc trưng
        """
        if self.column_transformer is None:
            raise ValueError("FeatureExtractor chưa được fit")

        return self.column_transformer.transform(df)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Học và chuyển đổi dữ liệu

        Args:
            df: DataFrame chứa dữ liệu
            y: Series chứa biến mục tiêu (tùy chọn)

        Returns:
            Ma trận đặc trưng
        """
        return self.fit(df, y).transform(df)

    def encode_labels(self, y: pd.Series, column_name: str) -> np.ndarray:
        """
        Mã hóa nhãn

        Args:
            y: Series chứa nhãn
            column_name: Tên cột nhãn

        Returns:
            Mảng nhãn đã mã hóa
        """
        if column_name not in self.label_encoders:
            encoder = LabelEncoder()
            self.label_encoders[column_name] = encoder
            return encoder.fit_transform(y)
        else:
            return self.label_encoders[column_name].transform(y)

    def decode_labels(self, y_encoded: np.ndarray, column_name: str) -> np.ndarray:
        """
        Giải mã nhãn

        Args:
            y_encoded: Mảng nhãn đã mã hóa
            column_name: Tên cột nhãn

        Returns:
            Mảng nhãn gốc
        """
        if column_name not in self.label_encoders:
            raise ValueError(f"Chưa có label encoder cho cột {column_name}")

        return self.label_encoders[column_name].inverse_transform(y_encoded)

    def _get_feature_names(self) -> List[str]:
        """
        Lấy tên các đặc trưng

        Returns:
            Danh sách tên đặc trưng
        """
        if self.column_transformer is None:
            return []

        feature_names = []

        for name, transformer, column in self.column_transformer.transformers_:
            try:
                # Thử sử dụng get_feature_names_out trước
                if hasattr(transformer, 'get_feature_names_out'):
                    if isinstance(column, str):
                        col_names = [f"{column}_{feat}" for feat in transformer.get_feature_names_out()]
                    else:
                        col_names = [f"{col}_{feat}" for col in column for feat in transformer.get_feature_names_out()]
                    feature_names.extend(col_names)
                # Dự phòng cho get_feature_names cũ
                elif hasattr(transformer, 'get_feature_names'):
                    if isinstance(column, str):
                        col_names = [f"{column}_{feat}" for feat in transformer.get_feature_names()]
                    else:
                        col_names = [f"{col}_{feat}" for col in column for feat in transformer.get_feature_names()]
                    feature_names.extend(col_names)
                # Xử lý OneHotEncoder
                elif hasattr(transformer, 'categories_'):
                    if isinstance(column, list) and len(column) == 1:
                        col = column[0]
                        categories = transformer.categories_[0]
                        col_names = [f"{col}_{cat}" for cat in categories]
                        feature_names.extend(col_names)
                # Xử lý FunctionTransformer và các transformer khác không có get_feature_names
                else:
                    if isinstance(column, str):
                        feature_names.append(column)
                    else:
                        feature_names.extend(column)
            except Exception as e:
                # Ghi log lỗi nhưng không làm dừng tiến trình
                logger.warning(f"Lỗi khi lấy tên đặc trưng từ transformer {name}: {str(e)}")
                # Sử dụng tên cột gốc làm dự phòng
                if isinstance(column, str):
                    feature_names.append(column)
                elif isinstance(column, list):
                    feature_names.extend(column)

        return feature_names

    def get_feature_names(self) -> List[str]:
        """
        Lấy danh sách tên đặc trưng

        Returns:
            Danh sách tên đặc trưng
        """
        if self.feature_names is None:
            self.feature_names = self._get_feature_names()

        return self.feature_names

    def get_feature_importance(self, model, top_n: int = None) -> List[Dict[str, Any]]:
        """
        Lấy độ quan trọng của các đặc trưng từ mô hình

        Args:
            model: Mô hình đã được huấn luyện (phải có thuộc tính feature_importances_)
            top_n: Số lượng đặc trưng quan trọng nhất cần lấy

        Returns:
            Danh sách các đặc trưng quan trọng và giá trị tương ứng
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Mô hình không có thuộc tính feature_importances_")

        if self.feature_names is None:
            raise ValueError("Chưa có thông tin về tên đặc trưng")

        importances = model.feature_importances_

        if len(importances) != len(self.feature_names):
            logger.warning(
                f"Số lượng đặc trưng ({len(self.feature_names)}) không khớp với "
                f"số lượng độ quan trọng ({len(importances)})"
            )
            return []

        # Tạo danh sách độ quan trọng
        feature_importance = [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in zip(self.feature_names, importances)
        ]

        # Sắp xếp theo độ quan trọng giảm dần
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        # Lấy top N nếu được chỉ định
        if top_n is not None and top_n > 0:
            feature_importance = feature_importance[:top_n]

        return feature_importance