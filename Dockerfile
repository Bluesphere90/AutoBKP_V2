FROM python:3.10-slim

WORKDIR /app

# Cài đặt các thư viện cần thiết
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn
COPY . .

# Tạo thư mục volumes
RUN mkdir -p /app/volumes/data /app/volumes/models /app/volumes/config /app/volumes/logs

# Cấu hình biến môi trường
ENV PYTHONPATH=/app
ENV DATA_PATH=/app/volumes/data
ENV MODELS_PATH=/app/volumes/models
ENV CONFIG_PATH=/app/volumes/config
ENV LOGS_PATH=/app/volumes/logs

# Chạy ứng dụng
CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]