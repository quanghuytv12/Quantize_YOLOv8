import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import onnxruntime

# Tải mô hình ONNX YOLOv8
input_model_path = "trained_model.onnx"
model = onnx.load(input_model_path)

# Thực hiện suy luận hình dạng
model_inferred = SymbolicShapeInference.infer_shapes(model)

# Lưu lại mô hình đã suy luận hình dạng
onnx.save(model_inferred, "trained_model_inferred.onnx")

# Bước 3: Tối ưu hóa mô hình (tùy chọn)
sess_option = onnxruntime.SessionOptions()
sess_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
session = onnxruntime.InferenceSession("trained_model_inferred.onnx", sess_option)

# (Tùy chọn) Bước 4: Lưu mô hình tối ưu hóa
onnx.save(model_inferred, "trained_model_optimized.onnx")
print("Model save successfully")