---
title: Ecg Ai Flask
emoji: "🫀"
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Hệ thống AI chẩn đoán điện tâm đồ ECG

Ứng dụng Flask hỗ trợ:
- phân tích ECG bằng AI
- xem lịch sử chẩn đoán
- báo cáo và đánh giá mô hình
- chạy local bằng Docker mà không cần cài Python, TensorFlow hay môi trường ảo

## Link server

Server đang chạy tại:

`https://zn3004-ecg-ai-flask.hf.space`

## Clone dự án

```bash
git clone https://github.com/vonguyen3004/XDHTChuanDoanBenhLyTimMach.git
cd XDHTChuanDoanBenhLyTimMach
```

## Chạy nhanh bằng Docker

Yêu cầu duy nhất:
- đã cài `Docker Desktop`

Sau khi clone dự án, mở terminal tại thư mục dự án và chạy:

```bash
docker compose up --build
```

Sau khi container khởi động xong, mở:

`http://localhost:7860`

Với cách này:
- Flask app chạy trong container `web`
- MongoDB chạy trong container `mongo`
- không cần cài thêm Python package hay tạo virtual environment

## Dừng hệ thống

```bash
docker compose down
```

Nếu muốn xóa cả dữ liệu Mongo local:

```bash
docker compose down -v
```

## Ghi chú

- Khi chạy local bằng Docker Compose, ứng dụng tự dùng MongoDB local trong container:
  `mongodb://mongo:27017`
- Khi deploy Hugging Face Space, app dùng biến môi trường `MONGO_URI`
