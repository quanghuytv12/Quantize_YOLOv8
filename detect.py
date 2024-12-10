from ultralytics import YOLO
import cv2

# Load mô hình YOLOv8 (ví dụ: yolov8n.pt, yolov8s.pt, v.v.)
model = YOLO("model.pt")

# # Đọc video từ webcam hoặc từ file
# video_path = 0  # 0 để sử dụng webcam hoặc thay bằng đường dẫn file video
# cap = cv2.VideoCapture(video_path)

# Đường dẫn tới hình ảnh
image_path = "frame_165.jpg"

# Dự đoán trên hình ảnh
results = model(image_path)

# Hiển thị kết quả
results[0].show()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Dự đoán (detection) trên từng frame
#     results = model(frame)
#
#     # Vẽ kết quả lên frame
#     annotated_frame = results[0].plot()
#
#     # Hiển thị frame đã annotate
#     cv2.imshow("YOLOv8 Detection", annotated_frame)
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# # Giải phóng tài nguyên
# cap.release()
# cv2.destroyAllWindows()
