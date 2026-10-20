# PCB-detection

## Chạy ứng dụng nhận diện trên máy local

Ứng dụng web dùng Streamlit để tải ảnh PCB từ máy tính, chạy mô hình YOLO đã huấn luyện
(`model/best.pt`) và hiển thị các hộp nhận diện cùng tên lớp, độ tin cậy và tọa độ.

1. Tạo và kích hoạt môi trường Python ảo (khuyến nghị).
2. Cài dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Từ thư mục gốc repo, chạy:

   ```bash
   streamlit run app/app.py
   ```

Trình duyệt sẽ mở trang ứng dụng. Chọn ảnh JPG, JPEG, PNG, BMP hoặc WEBP để xem kết quả.
Mô hình chạy trên máy local; ngưỡng tin cậy và IoU có thể điều chỉnh ở thanh bên.
