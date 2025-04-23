"""
Các công cụ đánh giá hiệu suất mô hình
- Tính các chỉ số đánh giá cho mô hình phân loại
- Vẽ đồ thị đánh giá
- Cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   average: str = 'weighted') -> Dict[str, Any]:
    """
    Đánh giá hiệu suất của mô hình phân loại

    Args:
        model: Mô hình cần đánh giá
        X_test: Dữ liệu kiểm tra
        y_test: Nhãn thực tế
        class_names: Tên các lớp
        average: Phương pháp tính trung bình cho các chỉ số

    Returns:
        Dict chứa các chỉ số đánh giá
    """
    # Dự đoán
    y_pred = model.predict(X_test)

    # Lấy xác suất nếu có
    try:
        y_prob = model.predict_proba(X_test)
        has_probabilities = True
    except:
        has_probabilities = False

    # Tính các chỉ số đánh giá cơ bản
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

    # Tạo báo cáo phân loại
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)

    # Kết quả
    result = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y_test)
    }

    # Thêm AUC nếu có xác suất
    if has_probabilities and len(np.unique(y_test)) == 2:  # binary classification
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        auc_score = auc(fpr, tpr)
        result['auc'] = float(auc_score)

        # Thêm thông tin đường cong ROC
        result['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }

        # Thêm thông tin đường cong Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob[:, 1])
        avg_precision = average_precision_score(y_test, y_prob[:, 1])

        result['pr_curve'] = {
            'precision': precision_curve.tolist(),
            'recall': recall_curve.tolist(),
            'average_precision': float(avg_precision)
        }

    return result


def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                         n_splits: int = 5, random_state: int = 42,
                         scoring: str = 'f1_weighted') -> Dict[str, Any]:
    """
    Thực hiện cross-validation cho mô hình

    Args:
        model: Mô hình cần đánh giá
        X: Dữ liệu
        y: Nhãn
        n_splits: Số lượng fold
        random_state: Hạt giống ngẫu nhiên
        scoring: Chỉ số đánh giá

    Returns:
        Dict chứa kết quả cross-validation
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Thực hiện cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return {
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'n_splits': n_splits,
        'scoring': scoring
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Tính ma trận nhầm lẫn

    Args:
        y_true: Nhãn thực tế
        y_pred: Nhãn dự đoán
        class_names: Tên các lớp

    Returns:
        Dict chứa ma trận nhầm lẫn và thông tin liên quan
    """
    # Tính ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)

    # Nếu không có tên lớp, sử dụng tên mặc định
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Tính các chỉ số từ ma trận nhầm lẫn
    n_classes = cm.shape[0]
    total_samples = np.sum(cm)

    # Tính độ chính xác từng lớp
    class_accuracy = {}
    for i in range(n_classes):
        if np.sum(cm[i, :]) > 0:
            class_accuracy[class_names[i]] = float(cm[i, i] / np.sum(cm[i, :]))
        else:
            class_accuracy[class_names[i]] = 0.0

    # Tính tỷ lệ dự đoán đúng và sai
    true_predictions = np.sum(np.diag(cm))
    false_predictions = total_samples - true_predictions

    accuracy = float(true_predictions / total_samples) if total_samples > 0 else 0.0
    error_rate = float(false_predictions / total_samples) if total_samples > 0 else 0.0

    return {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'class_accuracy': class_accuracy,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'total_samples': int(total_samples)
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          figsize: Tuple[int, int] = (10, 8),
                          cmap: str = 'Blues', normalize: bool = False) -> plt.Figure:
    """
    Vẽ ma trận nhầm lẫn

    Args:
        cm: Ma trận nhầm lẫn
        class_names: Tên các lớp
        figsize: Kích thước hình
        cmap: Bảng màu
        normalize: Chuẩn hóa giá trị

    Returns:
        Matplotlib Figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    return plt.gcf()


def plot_feature_importance(feature_importance: List[Dict[str, Any]],
                            top_n: int = 20,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Vẽ biểu đồ độ quan trọng của đặc trưng

    Args:
        feature_importance: Danh sách các đặc trưng và độ quan trọng
        top_n: Số lượng đặc trưng quan trọng nhất cần hiển thị
        figsize: Kích thước hình

    Returns:
        Matplotlib Figure
    """
    # Sắp xếp theo độ quan trọng giảm dần
    sorted_features = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

    # Lấy top N
    if top_n is not None and top_n > 0:
        sorted_features = sorted_features[:top_n]

    # Tạo danh sách tên và giá trị
    features = [item['feature'] for item in sorted_features]
    importances = [item['importance'] for item in sorted_features]

    # Vẽ biểu đồ
    plt.figure(figsize=figsize)
    plt.barh(range(len(features)), importances, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()

    return plt.gcf()