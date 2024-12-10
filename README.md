# Quantize YOLO Model to ONNX Format

Đây là dự án để quantize các mô hình YOLO vào định dạng ONNX.

## 📁 Cấu trúc thư mục

- `convert.py`: Script để chuyển đổi từ định dạng `.pt` sang `.onnx`.
- `pre_processing.py`: Tiền xử lý mô hình trước khi quantize.
- `dynamic_quantization_default.py`: Quantize dynamic với cài đặt mặc định.
- `dynamic_quantization_full.py`: Quantize dynamic với các cài đặt đầy đủ.
- `onnx_runtime.py`: Script để chạy inference sử dụng ONNX Runtime, được lấy từ ví dụ của ONNX Runtime.
