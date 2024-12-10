import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
import time


class YOLOv8:
    def __init__(self, onnx_model, input_file, confidence_thres, iou_thres, output_file):
        self.onnx_model = onnx_model
        self.input_file = input_file
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.output_file = output_file

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml("data.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, frame):
        self.img = frame
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def postprocess(self, input_image, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)
        return input_image

    def save_result(self, output_frame):
        return output_frame

    def process_video(self):
        # Open the video file
        cap = cv2.VideoCapture(self.input_file)
        if not cap.isOpened():
            print(f"Error: Couldn't open video {self.input_file}")
            return

        # Get video information
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:  # Ensure fps is not zero to avoid division by zero
            print("Error: FPS is zero, unable to process video.")
            return

        print(frame_width, frame_height)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, fps, (frame_width, frame_height))

        # Set the desired FPS (for example, we can try to process the video at 2x speed)
        desired_fps = 60  # Adjust as necessary
        frame_interval = max(int(fps / desired_fps), 1)  # Ensure frame_interval is never zero

        cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Video', frame_width, frame_height)

        start_time = time.time()  # Start time for FPS calculation

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure the frame is the correct size
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Process every nth frame
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_count % frame_interval == 0:
                # Preprocess the frame for inference
                img_data = self.preprocess(frame)

                # Run inference
                outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

                # Perform post-processing and get the result frame
                output_frame = self.postprocess(frame, outputs)

                # Write the output frame to the video
                out.write(output_frame)

                # Calculate FPS
                end_time = time.time()
                processing_time = end_time - start_time
                fps_display = 1.0 / processing_time
                start_time = end_time  # Update start time for next frame

                # Display FPS on the frame
                fps_text = f"FPS: {fps_display:.2f}"
                cv2.putText(output_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Show the output frame in a window
                cv2.imshow('Processed Video', output_frame)

                # Wait for a key press for 1ms (this helps control FPS)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print(f"Video processing complete. Output saved to {self.output_file}")

    def main(self):
        # Create an inference session using the ONNX model
        self.session = ort.InferenceSession(self.onnx_model, providers=["CPUExecutionProvider"])
        self.model_inputs = self.session.get_inputs()
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Check if the input is a video or image and process accordingly
        if self.input_file.lower().endswith(('.mp4', '.avi', '.mov')):
            self.process_video()
        else:
            self.process_image()


if __name__ == "__main__":
    # Parse arguments from CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--input", type=str, default="input_image.jpg", help="Path to input image or video.")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Path to output image or video.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # Create an instance of the YOLOv8 class with the specified arguments
    detection = YOLOv8(args.model, args.input, args.conf_thres, args.iou_thres, args.output)

    # Perform object detection on the input (either image or video) and save the output
    detection.main()
