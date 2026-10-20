# Báo Cáo Triển Khai & Ứng Dụng (Deployment Documentation)

Tài liệu này tổng hợp cấu hình inference pipeline, ứng dụng web Streamlit và hướng dẫn triển khai cho mô hình YOLO26n Object Detection.

## 1. Inference Pipeline & Post-processing
Mã nguồn xử lý: `app/app.py` (và `src/inference.py`)

Quy trình nhận diện và xử lý kết quả:
1. **Tiền xử lý:** Nhận ảnh đầu vào từ người dùng (JPG, PNG, BMP, WEBP), chuẩn hóa kênh màu RGB và đưa về kích thước 640×640 px theo chuẩn YOLO.
2. **Suy luận (Inference):** Nạp trọng số YOLO26n tại `model/best.pt`, dự đoán bounding box, nhãn linh kiện và điểm tin cậy (confidence score) với ngưỡng tùy chỉnh.
3. **Hậu xử lý (Post-processing):**
   - Lọc bỏ dự đoán có độ tin cậy thấp hơn ngưỡng quy định.
   - Áp dụng Non-Maximum Suppression (IoU threshold) để khử trùng lặp bounding box.
   - Tự động thống kê và đếm số lượng linh kiện tìm được theo từng class.

## 2. Cấu Trúc Ứng Dụng Web (Streamlit Application)
Giao diện ứng dụng được tổ chức thành 4 phân hệ chính:
- **Trang chủ:** Giới thiệu mục đích hệ thống, hiển thị thông số tổng quan và danh mục 10 linh kiện kèm ký hiệu mạch điện tử.
- **Nhận diện:** Tải ảnh bo mạch lên, thực hiện suy luận thời gian thực, hiển thị ảnh trực quan, bảng chi tiết tọa độ và cho phép tải ảnh kết quả về máy.
- **Hiệu suất mô hình:** Trình bày ma trận nhầm lẫn (confusion matrix), đồ thị hàm mất mát và bảng chỉ số mAP50, mAP50-95 theo từng linh kiện.
- **Gallery mẫu:** Xem nhanh tập kết quả nhận diện mẫu trên tập kiểm thử (test set).

### Ảnh demo ứng dụng

Ảnh dưới đây minh họa ảnh PCB đầu vào và kết quả nhận diện do YOLO26n tạo ra:

<table>
  <tr>
    <th>Ảnh PCB đầu vào</th>
    <th>Kết quả nhận diện</th>
  </tr>
  <tr>
    <td align="center"><img src="../assest/demo/demo_input.jpg" alt="Ảnh PCB đầu vào" width="400"></td>
    <td align="center"><img src="../assest/demo/demo_output.png" alt="Ảnh PCB sau khi YOLO26n nhận diện linh kiện" width="400"></td>
  </tr>
</table>

Ảnh chụp giao diện cho thấy kết quả trên trang nhận diện và thông tin chi tiết của từng linh kiện:

<table>
  <tr>
    <td align="center"><img src="../assest/demo/screenshot_detect_preview.png" alt="Giao diện nhận diện với ảnh gốc và ảnh kết quả" width="100%"></td>
    <td align="center"><img src="../assest/demo/screenshot_detect_details.png" alt="Thống kê và bảng chi tiết kết quả nhận diện" width="100%"></td>
  </tr>
</table>

## 3. Hướng Dẫn Cài Đặt & Triển Khai (Deployment)
1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Khởi chạy ứng dụng cục bộ:**
   ```bash
   streamlit run app/app.py
   ```
   Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`.

## 4. Tài Nguyên & Kết Quả (Assets)
- **Tập tin trọng số mô hình:** `model/best.pt`
- **Ảnh kết quả mẫu:** Lưu tại `assest/pred/` (gồm 20 ảnh dự đoán thực tế)
- **Biểu đồ đánh giá:** Lưu tại `assest/graph/` (đồ thị training curve, confusion matrix)
- **Ảnh chụp demo & sơ đồ:** Lưu tại `assest/demo/` và `assest/architecture/`
