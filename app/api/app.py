#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Service cho dự án phân loại HachToan và MaHangHoa
- Cung cấp endpoints để huấn luyện và dự đoán
- Quản lý mô hình cho nhiều khách hàng
"""

import os
import sys
from pathlib import Path
import logging
import json
import shutil
import tempfile
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, Query, BackgroundTasks, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import các module cần thiết
from app.scripts.validate_data import validate_data
from app.scripts.run_training import run_complete_training
import app.scripts.train as train_module
from app.scripts.run_prediction import run_prediction_from_file, run_prediction_from_json
from app.config import config_manager, path_manager, constants
from app.config.utils import save_metadata, load_metadata

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename='api.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('api')

# Khởi tạo FastAPI app
app = FastAPI(
    title="AutoBKP API",
    description="API Service cho phân loại HachToan và MaHangHoa",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả methods
    allow_headers=["*"],  # Cho phép tất cả headers
)

# Theo dõi các tác vụ nền
background_tasks = {}


# Định nghĩa các models
class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str
    customer_id: str
    timestamp: str


class PredictionRequest(BaseModel):
    MSTNguoiBan: str
    TenHangHoaDichVu: str
    additional_fields: Optional[Dict[str, Any]] = Field(default={})


class PredictionResponse(BaseModel):
    HachToan: Optional[str] = None
    MaHangHoa: Optional[str] = None
    HachToan_Probability: Optional[float] = None
    MaHangHoa_Probability: Optional[float] = None
    is_outlier: bool = False
    outlier_score: Optional[float] = None
    warnings: List[str] = []
    timestamp: str


class CustomerConfig(BaseModel):
    column_config: Optional[Dict[str, Any]] = None
    preprocessing_config: Optional[Dict[str, Any]] = None
    model_configuration: Optional[Dict[str, Any]] = None  # Đổi tên từ model_config
    training_config: Optional[Dict[str, Any]] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    customer_id: str
    task_type: str
    start_time: str
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Hàm helper cho background tasks
async def train_model_task(task_id: str, customer_id: str, file_path: str):
    """
    Tác vụ nền cho việc huấn luyện mô hình

    Args:
        task_id: ID của tác vụ
        customer_id: ID của khách hàng
        file_path: Đường dẫn đến file dữ liệu
    """
    try:
        background_tasks[task_id]["status"] = "processing"

        # Chạy tiến trình huấn luyện
        result = run_complete_training(customer_id, file_path)

        # Cập nhật trạng thái
        background_tasks[task_id]["status"] = "completed" if result["status"] == "success" else "failed"
        background_tasks[task_id]["end_time"] = datetime.now().isoformat()
        background_tasks[task_id]["result"] = result

        # Xóa file tạm nếu cần
        if file_path.startswith(tempfile.gettempdir()):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Không thể xóa file tạm {file_path}: {str(e)}")

    except Exception as e:
        logger.exception(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        background_tasks[task_id]["status"] = "failed"
        background_tasks[task_id]["end_time"] = datetime.now().isoformat()
        background_tasks[task_id]["error"] = str(e)


# API Endpoints
@app.get("/")
async def root():
    """Endpoint kiểm tra trạng thái API"""
    return {"status": "OK", "message": "AutoBKP API Service is running"}


@app.get("/api/customers")
async def get_customers():
    """Lấy danh sách khách hàng"""
    try:
        customers = config_manager.list_customers()
        return {"status": "success", "customers": customers}
    except Exception as e:
        logger.exception(f"Lỗi khi lấy danh sách khách hàng: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy danh sách khách hàng: {str(e)}"
        )


@app.post("/api/customers/{customer_id}/config")
async def update_customer_config(customer_id: str, config: CustomerConfig):
    """Cập nhật cấu hình cho khách hàng"""
    try:
        # Chuyển đổi từ model Pydantic sang dict
        config_dict = config.model_dump(exclude_none=True)

        # Sửa lại tên trường nếu cần
        if 'model_configuration' in config_dict:
            config_dict['model_config'] = config_dict.pop('model_configuration')

        # Cập nhật cấu hình
        updated_config = config_manager.update_customer_config(customer_id, config_dict)

        return {
            "status": "success",
            "message": f"Đã cập nhật cấu hình cho khách hàng {customer_id}",
            "config": updated_config
        }
    except Exception as e:
        logger.exception(f"Lỗi khi cập nhật cấu hình: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi cập nhật cấu hình: {str(e)}"
        )


@app.get("/api/customers/{customer_id}/config")
async def get_customer_config(customer_id: str):
    """Lấy cấu hình của khách hàng"""
    try:
        config = config_manager.get_customer_config(customer_id)
        # Đổi tên trường nếu trả về cho UI
        if 'model_config' in config:
            config['model_configuration'] = config.pop('model_config')
        return {"status": "success", "config": config}
    except Exception as e:
        logger.exception(f"Lỗi khi lấy cấu hình: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy cấu hình: {str(e)}"
        )


@app.post("/api/customers/{customer_id}/validate")
async def validate_customer_data(
        customer_id: str,
        file: UploadFile = File(...),
):
    """Kiểm tra tính hợp lệ của dữ liệu"""
    try:
        # Lưu file tạm thời
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file_path = temp_file.name
        temp_file.close()

        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Thực hiện kiểm tra
        validation_result = validate_data(temp_file_path, customer_id)

        # Xóa file tạm
        try:
            os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Không thể xóa file tạm {temp_file_path}: {str(e)}")

        return validation_result

    except Exception as e:
        logger.exception(f"Lỗi khi kiểm tra dữ liệu: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi kiểm tra dữ liệu: {str(e)}"
        )


@app.post("/api/customers/{customer_id}/train")
async def train_customer_model(
        background_task_manager: BackgroundTasks,  # Đổi tên tham số
        customer_id: str,
        file: UploadFile = File(...),
        incremental: bool = Form(False)
):
    """Huấn luyện mô hình cho khách hàng"""
    try:
        # Lưu file tạm thời
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file_path = temp_file.name
        temp_file.close()

        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Tạo ID cho tác vụ
        task_id = f"train_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Khởi tạo thông tin tác vụ
        task_info = {
            "task_id": task_id,
            "status": "pending",
            "customer_id": customer_id,
            "task_type": "incremental_training" if incremental else "training",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "result": None,
            "error": None
        }

        background_tasks[task_id] = task_info

        # Khởi động tác vụ nền
        background_task_manager.add_task(train_model_task, task_id, customer_id, temp_file_path)

        return TrainingResponse(
            task_id=task_id,
            status="pending",
            message=f"Đã bắt đầu huấn luyện mô hình cho khách hàng {customer_id}",
            customer_id=customer_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.exception(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi huấn luyện mô hình: {str(e)}"
        )


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Lấy trạng thái của tác vụ"""
    if task_id not in background_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy tác vụ với ID {task_id}"
        )

    task_info = background_tasks[task_id]

    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        customer_id=task_info["customer_id"],
        task_type=task_info["task_type"],
        start_time=task_info["start_time"],
        end_time=task_info["end_time"],
        result=task_info["result"],
        error=task_info["error"]
    )


@app.post("/api/customers/{customer_id}/predict")
async def predict_single(customer_id: str, request: PredictionRequest):
    """Dự đoán cho một mẫu đơn lẻ"""
    try:
        # Chuẩn bị dữ liệu đầu vào
        input_data = {
            "MSTNguoiBan": request.MSTNguoiBan,
            "TenHangHoaDichVu": request.TenHangHoaDichVu
        }

        # Thêm các trường bổ sung nếu có
        if request.additional_fields:
            input_data.update(request.additional_fields)

        # Thực hiện dự đoán
        result = run_prediction_from_json(customer_id, input_data)

        if result["status"] != "success":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Lỗi khi dự đoán: {result.get('error', 'Unknown error')}"
            )

        # Lấy kết quả dự đoán
        prediction = result["prediction"]

        return PredictionResponse(
            HachToan=prediction.get("prediction", {}).get("HachToan"),
            MaHangHoa=prediction.get("prediction", {}).get("MaHangHoa"),
            HachToan_Probability=prediction.get("probabilities", {}).get("HachToan"),
            MaHangHoa_Probability=prediction.get("probabilities", {}).get("MaHangHoa"),
            is_outlier=prediction.get("outlier_warning", False),
            outlier_score=prediction.get("outlier_score"),
            warnings=prediction.get("warnings", []),
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Lỗi khi dự đoán: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi dự đoán: {str(e)}"
        )


@app.post("/api/customers/{customer_id}/predict-batch")
async def predict_batch(
        background_task_manager: BackgroundTasks,  # Đổi tên tham số
        customer_id: str,
        file: UploadFile = File(...),
        model_version: str = Form("latest")
):
    """Dự đoán hàng loạt từ file CSV"""
    try:
        # Lưu file tạm thời
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file_path = temp_file.name
        temp_file.close()

        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Tạo ID cho tác vụ
        task_id = f"predict_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Khởi tạo thông tin tác vụ
        task_info = {
            "task_id": task_id,
            "status": "pending",
            "customer_id": customer_id,
            "task_type": "batch_prediction",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "result": None,
            "error": None
        }

        background_tasks[task_id] = task_info

        # Khởi động tác vụ nền
        async def predict_batch_task(task_id: str, customer_id: str, file_path: str, model_version: str):
            try:
                background_tasks[task_id]["status"] = "processing"

                # Thực hiện dự đoán
                result = run_prediction_from_file(customer_id, file_path, version=model_version)

                # Cập nhật trạng thái
                background_tasks[task_id]["status"] = "completed" if result["status"] == "success" else "failed"
                background_tasks[task_id]["end_time"] = datetime.now().isoformat()
                background_tasks[task_id]["result"] = result

                # Xóa file tạm
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Không thể xóa file tạm {file_path}: {str(e)}")

            except Exception as e:
                logger.exception(f"Lỗi khi dự đoán hàng loạt: {str(e)}")
                background_tasks[task_id]["status"] = "failed"
                background_tasks[task_id]["end_time"] = datetime.now().isoformat()
                background_tasks[task_id]["error"] = str(e)

        background_task_manager.add_task(predict_batch_task, task_id, customer_id, temp_file_path, model_version)

        return {
            "task_id": task_id,
            "status": "pending",
            "message": f"Đã bắt đầu dự đoán hàng loạt cho khách hàng {customer_id}",
            "customer_id": customer_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.exception(f"Lỗi khi dự đoán hàng loạt: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi dự đoán hàng loạt: {str(e)}"
        )


@app.get("/api/customers/{customer_id}/models")
async def get_customer_models(customer_id: str):
    """Lấy thông tin về các mô hình của khách hàng"""
    try:
        model_path = path_manager.get_customer_model_path(customer_id)

        if not os.path.exists(model_path):
            return {"status": "success", "models": [], "message": "Chưa có mô hình nào"}

        # Kiểm tra các loại mô hình
        models_info = {}

        # Kiểm tra mô hình HachToan
        hachtoan_path = os.path.join(model_path, "hachtoan")
        if os.path.exists(hachtoan_path):
            hachtoan_versions = [f.replace("model_", "").replace(".joblib", "")
                                 for f in os.listdir(hachtoan_path)
                                 if f.startswith("model_") and f != "model_latest.joblib"]

            models_info["hachtoan"] = {
                "available": True,
                "versions": sorted(hachtoan_versions),
                "latest_version": max(hachtoan_versions) if hachtoan_versions else None
            }
        else:
            models_info["hachtoan"] = {"available": False}

        # Kiểm tra mô hình MaHangHoa
        mahanghua_path = os.path.join(model_path, "mahanghua")
        if os.path.exists(mahanghua_path):
            mahanghua_versions = [f.replace("model_", "").replace(".joblib", "")
                                  for f in os.listdir(mahanghua_path)
                                  if f.startswith("model_") and f != "model_latest.joblib"]

            models_info["mahanghua"] = {
                "available": True,
                "versions": sorted(mahanghua_versions),
                "latest_version": max(mahanghua_versions) if mahanghua_versions else None
            }
        else:
            models_info["mahanghua"] = {"available": False}

        # Lấy metadata mô hình mới nhất
        metadata_path = os.path.join(model_path, "metadata", "model_metadata_latest.json")
        metadata = None
        if os.path.exists(metadata_path):
            metadata = load_metadata(metadata_path)

        return {
            "status": "success",
            "customer_id": customer_id,
            "models": models_info,
            "metadata": metadata
        }

    except Exception as e:
        logger.exception(f"Lỗi khi lấy thông tin mô hình: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy thông tin mô hình: {str(e)}"
        )


@app.get("/api/customers/{customer_id}/results/{filename}")
async def get_prediction_result(customer_id: str, filename: str):
    """Lấy file kết quả dự đoán"""
    try:
        results_path = path_manager.get_customer_data_path(customer_id, "results")
        file_path = os.path.join(results_path, filename)

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Không tìm thấy file {filename}"
            )

        return FileResponse(
            file_path,
            media_type="text/csv",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Lỗi khi lấy file kết quả: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy file kết quả: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)