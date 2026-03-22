# HƯỚNG DẪN KIỂM THỬ HỆ THỐNG

## 1. KIỂM THỬ FILE CSV (Upload Mẫu)

### Yêu cầu:
- File: `data/upload_sample.csv`
- Format: 1 hàng dữ liệu ECG chuẩn hóa (0-1)
- Số điểm: 188 giá trị (hoặc thay đổi tùy theo mô hình)

### Các bước:

1. **Vào trang Upload**
   - URL: http://127.0.0.1:5000/upload
   
2. **Chế độ: CSV**
   - Chọn radio button "CSV (timesteps x 12 leads)" 
   
3. **Nhập thông tin bệnh nhân**
   - Mã bệnh nhân: VD001
   - Họ tên: Nguyễn Văn An
   - Tuổi: 45
   - Giới tính: Nam
   - Số điện thoại: 0901234567
   - Địa chỉ: Cần Thơ, VN
   - Triệu chứng: Đau ngực nhẹ
   
4. **Chọn file CSV**
   - Kéo thả hoặc nhấp vào dropzone
   - Chọn `data/upload_sample.csv`
   
5. **Xem trước (Preview)**
   - Nhấp nút "🖥 XEM TRƯỚC AI"
   - Quan sát kết quả dự đoán: nhãn chính, xác suất, các bệnh khác
   
6. **Gửi**
   - Nhấp nút "📤 GỬI"
   - Chờ hệ thống xử lý (~2-5 giây)
   - Được chuyển hướng tới trang kết quả chi tiết
   
### Kết quả mong đợi:

Trang chi tiết hiển thị:
- ✓ Thông tin bệnh nhân
- ✓ Biểu đồ sóng ECG (Chart.js)
- ✓ **KẾT QUẢ CHẨN ĐOÁN** (tiêu đề xanh lá cây)
  - Nhãn chính: VD: "PVC - Ngoại tâm thu thất"
  - Thẻ bệnh: THẺ BỆNH PVC (chip xanh lá cây)
  - Mô tả chi tiết (dòng lẻ)
- ✓ **Nhịp tim hiện tại** (BPM card)
  - Giá trị BPM
  - Trạng thái: ỔN ĐỊNH / CẦN THEO DÕI / BẤT THƯỜNG (màu code)
- ✓ **Danh sách dự đoán**
  - Tất cả 6 loại bệnh với xác suất (%)
  - Thanh tiến trình cho từng bệnh

---

## 2. KIỂM THỬ FILE .HEA + .DAT (PTB-XL Dataset)

### Yêu cầu:
- File .HEA: `data/ptbxl/records100/00000/00001_lr.hea`
- File .DAT: `data/ptbxl/records100/00000/00001_lr.dat`
- Format: WFDB (MIT-BIH), 12 lead, 100 Hz, 1000 samples

### Các bước:

1. **Vào trang Upload**
   - URL: http://127.0.0.1:5000/upload
   
2. **Chế độ: WFDB**
   - Chọn radio button "WFDB (.hea + .dat)"
   - Khu vực upload sẽ đổi sang 2 input field
   
3. **Nhập thông tin bệnh nhân**
   - (Như bước 3 ở mục CSV, nhưng có thể thay đổi ID)
   
4. **Chọn cặp file .HEA + .DAT**
   - Input 1: Chọn file `.hea`
     - Tìm: `data/ptbxl/records100/00000/00001_lr.hea`
   - Input 2: Chọn file `.dat`
     - Tìm: `data/ptbxl/records100/00000/00001_lr.dat`
   
5. **Xem trước (Preview)**
   - Nhấp nút "🖥 XEM TRƯỚC AI"
   - Observe thông tin được parse từ .HEA:
     - Sampling rate: 100 Hz
     - Số mẫu: 1000
     - Lead: I, II, III, AVR, AVL, AVF, V1-V6
   
6. **Gửi**
   - Nhấp nút "📤 GỬI"
   - Hệ thống sẽ:
     - Parse file .HEA + .DAT
     - Trích xuất dữ liệu ECG (12 lead)
     - Tiền xử lý (chuẩn hóa, downsample 1000→256 sample từ lead II)
     - Chạy CNN inference
     - Lưu vào MongoDB
   
### Kết quả mong đợi:

Tương tự như CSV, nhưng:
- ✓ Xem trước (Preview) hiển thị: "Inference Meta: source_ref=/tmp/xxxxx/00001_lr"
- ✓ Trang chi tiết có thêm biểu đồ ECG có 12 lead (hoặc chỉ lead II tùy UI)
- ✓ Recording name hiển thị: "00001_lr.hea + 00001_lr.dat"

---

## 3. KIỂM THỬ LỊCH SỬ VÀ IN BÁOCÁO

### Yêu cầu:
- Sau khi upload ít nhất 1 file (CSV hoặc WFDB)

### Các bước:

1. **Xem Lịch sử**
   - Menu: "📊 LỊCH SỬ" hoặc "History"
   - Hiển thị danh sách tất cả kết quả phân tích
   - Các cột: Bệnh nhân, Ngày/Giờ, Kết quả chẩn đoán
   
2. **In Báo cáo**
   - Tại trang chi tiết (view.html), nhấp nút "📄 IN"
   - Sẽ mở PDF hoặc print dialog
   - PDF chứa: Thông tin BN, kết quả chẩn đoán, biểu đồ ECG

---

## 4. KIỂM THỬ NGƯỠNG (THRESHOLD)

### Yêu cầu:

### Các bước:

1. **Threshold chung** (Range slider)
   - Giá trị mặc định: 0.5
   - Điều chỉnh từ 0.1 → 0.9
   - Ảnh hưởng: Tất cả bệnh dùng ngưỡng này
   
2. **Threshold map** (JSON, tùy chọn)
   - Format: `{"PAC": 0.35, "PVC": 0.4, "NORM": 0.6, ...}`
   - Nếu cung cấp, sẽ override threshold chung
   
3. **Test**
   - Upload file CSV/WFDB
   - Nhấp "Preview" để thấy kết quả với ngưỡng hiện tại
   - Điều chỉnh slider
   - Nhấp "Preview" lại để xem thay đổi

### Kết quả mong đợi:

- Ngưỡng cao (0.7-0.9): Ít khả năng phát hiện bệnh bệnh (bảo thủ)
- Ngưỡng thấp (0.1-0.3): Dễ phát hiện bệnh (nhạy cảm)
- Ngưỡng tối ưu (0.4-0.6): Cân bằng độ nhạy-độ đặc

---

## 5. KIỂM THỬ GỬI BẰNG MULTIPLE LEADS (Nếu có)

### (Tuỳ chọn - phần này dành cho phát triển thêm)

CSV format mở rộng:
```
sample_1,sample_2,...,sample_N
```

hoặc

```
I_1,I_2,...,I_N,II_1,II_2,...,II_N,...,V6_1,V6_2,...,V6_N
```

---

## 6. KIỂM THỬ LỖIƠI (ERROR HANDLING)

### Test case:
1. **Upload CSV không hợp lệ**
   - Tệp không phải CSV
   - CSV có cấu trúc sai
   - 👉 Kỳ vọng: Thông báo lỗi rõ ràng
   
2. **Upload WFDB thiếu file**
   - Chỉ chọn .HEA mà không chọn .DAT
   - 👉 Kỳ vọng: "Vui lòng chọn đủ 2 tệp .hea và .dat"
   
3. **Thông tin bệnh nhân không đủ**
   - Không nhập mã BN hoặc tên
   - 👉 Kỳ vọng: Form validation, nút Submit bị vô hiệu hóa
   
4. **Threshold map không phải JSON**
   - Nhập: `{not valid json}`
   - 👉 Kỳ vọng: Lỗi JSON parsing, thông báo warinng

---

## 7. TÓM TẮT CHECKPOINT VERIFY

| Feature | CSV | WFDB | Expected Result |
|---------|-----|------|-----------------|
| Upload form | ✓ | ✓ | Form rendered correctly |
| File selection | ✓ | ✓ | Files are picked |
| Patient info input | ✓ | ✓ | Data saved |
| Preview AI | ✓ | ✓ | Predictions shown (< 5s) |
| Inference | ✓ | ✓ | Results in DB |
| View detail page | ✓ | ✓ | All sections displayed |
| ECG chart | ✓ | ✓ | Chart.js renders |
| Diagnosis title | ✓ | ✓ | Green color (#34d399), glow |
| Diagnosis badge | ✓ | ✓ | THẺ BỆNH chip styled |
| BPM card | ✓ | ✓ | Gradient, status chip |
| History page | ✓ | ✓ | Records listed |
| Print/PDF | ✓ | ✓ | Report generated |

---

## 8. SCREENSHOT ĐỀ XUẤT CHO BÁO CÁO

Khi kiểm thử, hãy chụp lại:
1. Trang Upload (chế độ CSV + WFDB)
2. Preview AI (kết quả dự đoán)
3. Trang chi tiết (view.html) - đầy đủ
4. Trang Lịch sử (history.html)
5. PDF report (nếu có tính năng in)

---

## 9. LƯU Ý KHÁC

### Dữ liệu test:
- `data/upload_sample.csv` - CSV mẫu
- `data/ptbxl/records100/00000/` - PTB-XL sample records (00001_lr đến ~00010_lr)

### Mô hình sử dụng:
- CNN: `ml/cnn_model/cnn_model.keras`
- Classes: `NORM`, `PVC`, `PAC`, `LBBB`, `RBBB`, `BÌNH THƯỜNG`
- Sampling: 100 Hz (PTB-XL) → 256 samples (model input)

### Hiệu suất:
- Inference time: 1-2 giây/file (GPU) hoặc 2-5 giây (CPU)
- Database: MongoDB lưu lịch sử
- Server: Flask development mode (không dùng cho production)

---

**Ghi chú:** Hướng dẫn này dành cho báo cáo đồ án và kiểm thử chức năng hệ thống.
