import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType

# Tải mô hình ONNX YOLOv8
input_model_path = "model.onnx"
model = onnx.load(input_model_path)

# Thực hiện suy luận hình dạng
model_inferred = SymbolicShapeInference.infer_shapes(model)

# Lưu lại mô hình đã suy luận hình dạng
onnx.save(model_inferred, "model_inferred.onnx")
print("Inferred model save successfully\n")

# Bước 3: Tối ưu hóa mô hình (tùy chọn)
sess_option = onnxruntime.SessionOptions()
sess_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
session = onnxruntime.InferenceSession("model_inferred.onnx", sess_option)

# # (Tùy chọn) Bước 4: Lưu mô hình tối ưu hóa
onnx.save(model_inferred, "model_optimized.onnx")
print("Optimized model save successfully\n")

# Tải mô hình đã tối ưu
model_path = "model_optimized.onnx"
output_model_path = "model_dynamic_custom_quantized.onnx"

# # Áp dụng dynamic quantization (quantize weights thành int8)
# quantized_model = quantize_dynamic(
#     model_input=model_path,
#     model_output=output_model_path,                      # Đường dẫn tới mô hình đã tối ưu
#     weight_type=QuantType.QInt8,                            # Quantize trọng số thành int8
#     op_types_to_quantize=["Conv", "Gemm"],  # Tuỳ chọn: chỉ định các loại toán tử cần quantize (ví dụ: Conv, Gemm)
# )

quantized_model = quantize_dynamic(
    model_input= model_path,  # Đường dẫn mô hình đầu vào
    model_output=output_model_path, #Đường dẫn mô hình đầu ra
    # weight_type=QuantType.QInt8,  # Dữ liệu trọng số sẽ được lượng tử hóa thành int8
    weight_type=QuantType.QUInt8, #change QInt8 to QUInt8
    op_types_to_quantize=["Conv", "Gemm"],  # Lượng tử hóa các toán tử Conv và Gemm
    per_channel=True,  # Lượng tử hóa theo kênh
    reduce_range=True,  # Giảm phạm vi lượng tử hóa
    nodes_to_quantize=None,  # Lượng tử hóa tất cả các lớp
    nodes_to_exclude=None,  # Không loại trừ lớp nào
    use_external_data_format=False,  # Không dùng định dạng dữ liệu ngoài
    extra_options={
        "WeightSymmetric": True,  # Sử dụng đối xứng hóa cho trọng số
        "ActivationSymmetric": False,  # Không sử dụng đối xứng hóa cho kích hoạt
    }
)

# Lưu mô hình đã quantized
# onnx.save_model(quantized_model, output_model_path)
