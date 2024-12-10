import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Tải mô hình đã tối ưu
model_path = "trained_model.onnx"
output_model_path = "trained_model_quantized.onnx"

# Áp dụng dynamic quantization (quantize weights thành int8)
quantized_model = quantize_dynamic(
    model_input=model_path,
    model_output=output_model_path,                      # Đường dẫn tới mô hình đã tối ưu
    weight_type=QuantType.QInt8,                            # Quantize trọng số thành int8
    # op_types_to_quantize=["Conv", "Gemm"],  # Tuỳ chọn: chỉ định các loại toán tử cần quantize (ví dụ: Conv, Gemm)
)

# Lưu mô hình đã quantized
onnx.save_model(quantized_model, output_model_path)

