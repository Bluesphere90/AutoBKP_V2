FROM python:3.10-slim

# Cài đặt các gói build-essential và rust
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH"

WORKDIR /app

# Cài đặt maturin trước
RUN pip install --no-cache-dir maturin

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