# HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN BỆNH LÝ TIM MẠCH  

Dự án là một web‑app dùng Flask phía sau (backend) và HTML/JS phía trước (frontend), kết nối với MongoDB để lưu trữ các bản ghi ECG. Ngoài ra tích hợp mô hình Machine Learning để hỗ trợ gợi ý chẩn đoán dựa trên tín hiệu ECG.

Người dùng có thể upload file CSV chứa dãy mẫu ECG, xem danh sách/chi tiết, lọc theo thời gian, in hoặc xuất CSV, và xem đồ thị nhịp tim/chẩn đoán. Ứng dụng chạy trên localhost, có thể mở rộng để triển khai thực tế.

📋 MỤC ĐÍCH

- Giúp lưu trữ, quản lý và xem lại bản ghi ECG một cách trực quan.
- Cung cấp phân tích nhịp tim tự động và gợi ý chẩn đoán dựa trên mô hình học máy.
- Minh hoạ kiến trúc 3‑tier: frontend, backend, database, kèm theo minh hoạ tích hợp ML.

---

📊 BÁO CÁO TIẾN ĐỘ & CHỨC NĂNG CHÍNH

1. Frontend
- Trang dashboard với biểu đồ lượt tải mỗi ngày và số liệu tổng quan.
- Form upload kéo‑thả, lịch sự, kèm kéo/chọn file.
- Bảng lịch sử phân trang, tìm kiếm theo tên/ID/ngày, radio chọn khoảng thời gian, trường ngày tuỳ chọn.
- Nút in/ xuất CSV tại cuối bảng.
- Trang xem chi tiết ECG hiển thị biểu đồ, nhịp tim và dự đoán.
- Sử dụng Bootstrap/ AdminLTE, các tiện ích CSS để căn giữa, in hoa tự động, sidebar collapsible.

2. Backend
- Flask app xử lý routing, file upload, query MongoDB.
- API REST cơ bản phục vụ frontend (datalist gợi ý, export CSV, history lọc trang).
- Logic phân tích tín hiệu: chuẩn hoá, tính nhịp tim qua đỉnh, kiểm tra khoảng cách.
- Xử lý dữ liệu đầu vào, loại bỏ cột nhãn thừa nếu có.
- Tích hợp MongoDB; collection `ecgs` chứa các tài liệu gồm tên, timestamp, tín hiệu, kết quả ML.

3. Machine Learning
- Script Huấn luyện `ml/train_model.py` với RandomForestClassifier (50 cây).
- Dataset đầu vào: file CSV mỗi hàng là mẫu ECG; cột cuối cùng là nhãn lớp (0‑4 theo MIT‑BIH) hoặc dữ liệu tổng hợp.
- Trong môi trường test, mô hình đạt ~98‑99% accuracy trên tập kiểm tra giả. Với dữ liệu MIT‑BIH thực, độ chính xác thay đổi tùy bộ.
- Mô hình lưu dưới `backend/model.joblib` và được load khi app khởi động.

---

🏗️ KIẾN TRÚC DỰ ÁN

```
📦 XDHTCHUANDOANBENHLYTIMMACH/        # Hệ thống chuẩn đoán bệnh lý tim mạch
│
├── 📄 README.md                     # Tài liệu mô tả hệ thống
│
├── 📂 data/                         # Dữ liệu mẫu & test
│   └── upload_sample.csv            # File ECG mẫu để demo upload
│
├── 📂 tools/                        # Công cụ hỗ trợ xử lý dữ liệu
│   ├── inspect_dataset.py           # Kiểm tra cấu trúc dataset
│   └── regen_sample.py              # Tạo dữ liệu ECG giả lập
│
├── 📂 ml/                           # Machine Learning Module
│   ├── train_model.py               # Huấn luyện model
│   │
│   └── 📂 ECG_Diagnosis_System/     # Dataset MIT-BIH từ Kaggle
│       ├── mitbih_train.csv
│       └── mitbih_test.csv
│
├── 📂 backend/                      # Flask Web Application
│   ├── app.py                       # Server chính, logic xử lý
│   ├── requirements.txt             # Thư viện Python
│   ├── model.joblib                 # Tạo sau khi huấn luyện
│   │
│   ├── 📂 templates/                # Giao diện HTML (Jinja2)
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── upload.html
│   │   ├── list.html
│   │   ├── history.html
│   │   ├── view.html
│   │   └── print_ecg.html
│   │
│   └── 📂 static/                   # CSS, JS, hình ảnh
│       ├── 📂 css/
│       │   └── style.css
│       ├── 📂 js/
│       │   └── ecg.js
│       └── 📂 img/
│           └── favicon.ico
│
└── 📂 frontend/                     # Giao diện tĩnh độc lập
    ├── index.html
    └── script.js

Mỗi phần chịu trách nhiệm rõ ràng:
- `ml/`: xử lý dữ liệu và huấn luyện mô hình.
- `backend/`: phục vụ nội dung web và API, kết nối MongoDB.
- `frontend/`: nếu cần xây dựng giao diện tĩnh tách biệt.

---

🛠️ CÔNG NGHỆ SỬ DỤNG

- Ngôn ngữ: Python 3.11 (Flask, pandas, numpy, sklearn, pymongo).
- Framework web: Flask + Jinja2 templates.
- Cơ sở dữ liệu: MongoDB (pymongo).
- UI/JS: Bootstrap 5, AdminLTE theme, Chart.js, FontAwesome.
- ML: scikit-learn (RandomForestClassifier, 50 estimators), joblib lưu mô hình.
- Dữ liệu: tập MIT‑BIH (187 mẫu + nhãn) hoặc dữ liệu tổng hợp.
- Khác: venv, pandas, matplotlib (nếu cần).

---

🧠 Mô hình và dữ liệu

- Loại mô hình: Random Forest classifier (số cây 50). Mục tiêu phân loại 5 nhãn ECG cơ bản (bình thường, LBBB, RBBB, PVC, PAC).
- Dữ liệu đầu vào: CSV, mỗi hàng là vector tín hiệu ECG; cột cuối cùng chứa mã lớp (0‑4). Dữ liệu MIT‑BIH được cung cấp trong `ml/ECG_Diagnosis_System`.
- Chuẩn hoá: giá trị được chuẩn về trung bình 0, độ lệch chuẩn 1 trước khi đưa vào mô hình.
- Kết quả huấn luyện: độ chính xác trên tập kiểm tra (vd. chạy thử với dữ liệu giả) đạt khoảng 98‑99%. Khi dùng bộ MIT‑BIH thật có thể đạt khoảng 95‑99% tuỳ kích thước và tiền xử lý.
- Mô hình sau huấn luyện được lưu tại `backend/model.joblib` và load khi server khởi động.

---

🚀 Triển khai và mở rộng

- Có thể đóng gói bằng Docker, thêm xác thực người dùng, hoặc triển khai lên máy chủ thật.
- Mô hình dễ thay bằng bất kỳ bộ huấn luyện khác (XGBoost, NN) miễn sao xuất ra file joblib.
- Thêm API cho mobile/app và bảo mật CORS/SSL.

---

