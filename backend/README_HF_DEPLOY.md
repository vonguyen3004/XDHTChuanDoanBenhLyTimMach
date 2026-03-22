# Deploy lên Hugging Face Spaces

## 1. Cấu trúc thư mục
- Giữ nguyên backend/, static/, templates/, model, requirements.txt, app.py
- Đảm bảo các file model (.joblib, .keras) nằm đúng vị trí

## 2. File cấu hình Space
Tạo file `README.md` và `requirements.txt` ở thư mục gốc hoặc backend/.

## 3. Entrypoint
Entrypoint là `app.py` (Flask). Nếu Hugging Face yêu cầu file `app.py` ở thư mục gốc, hãy copy từ backend/ ra ngoài.

## 4. Lệnh deploy
```sh
pip install huggingface_hub
huggingface-cli login --token <TOKEN_CỦA_BẠN>
# Tạo Space mới trên web (chọn loại: "Other")
cd e:/XDHTChuanDoanBenhLyTimMach
huggingface-cli repo create <tên_space> --type=space --space-sdk=static
# Hoặc clone repo Space vừa tạo:
git clone https://huggingface.co/spaces/Zn3004/<tên_space>
cd <tên_space>
# Copy toàn bộ mã nguồn vào repo này
# Commit và push:
git add .
git commit -m "Deploy ECG AI Flask app"
git push
```

## 5. Lưu ý
- Nếu Flask không chạy được cổng 7860, hãy đặt biến môi trường `PORT` theo Hugging Face yêu cầu.
- Nếu cần, thêm file `app.py` ở thư mục gốc với nội dung import từ backend/app.py.
- Đảm bảo requirements.txt đầy đủ các thư viện.