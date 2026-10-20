# Báo cáo Huấn luyện & Đánh giá Mô hình YOLO26n (PCB Component Detection)

## 1. Class Names (Danh sách 10 lớp linh kiện)
0. `resistor` (Điện trở)
1. `capacitor` (Tụ điện)
2. `connector` (Đầu nối)
3. `ic` (Vi mạch IC)
4. `led` (Đèn LED)
5. `diode` (Điốt)
6. `transistor` (Tranzito)
7. `inductor` (Cuộn cảm)
8. `jumper` (Cầu nối)
9. `button` (Nút bấm)

## 2. Model Configuration (Cấu hình Mô hình)
- **Architecture:** YOLO26n (Nano)
- **Pretrained Weights:** `yolo26n.pt` (Transfer Learning từ tập COCO)
- **Input Image Size:** $640 	imes 640$ pixels
- **Model Weights Path:** `model/best.pt`

## 3. Training Configuration (Cấu hình Huấn luyện)
- **Dataset:** `./pcb_dataset_cleaned/data.yaml` (Tỷ lệ phân chia: 80% Train, 10% Val, 10% Test sau khi đã làm sạch bbox)
- **Epochs:** 100 (Có tích hợp `patience=20` để dừng sớm chống Overfitting)
- **Batch Size:** 16
- **Optimizer:** `AdamW`
- **Initial Learning Rate (lr0):** 0.001
- **Data Augmentations:** 
  - `mosaic = 1.0`
  - `mixup = 0.15`
  - `copy_paste = 0.3`
  - `degrees = 90.0`
  - `flipud = 0.5`, `fliplr = 0.5`

## 4. Evaluation Metrics (Chỉ số đánh giá)
- **Box Precision (P) & Recall (R):** Đạt ngưỡng tối ưu trên tập Validation và Test.
- **mAP50:** Độ chính xác trung bình ở ngưỡng IoU = 0.5.
- **mAP50-95:** Độ chính xác khắt khe trên toàn dải IoU từ 0.5 đến 0.95.

## 5. Assets & Prediction Samples (Kết quả trực quan)
Toàn bộ biểu đồ training, ma trận nhầm lẫn và mẫu dự đoán được lưu tại `assets/`:
- `training_curves.png` / `comprehensive_training_metrics.png`
- `confusion_matrix.png`
- `test_predictions_sample.png`
