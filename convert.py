from ultralytics import YOLO

model = YOLO(r"model.pt")

model.export(format='onnx')
