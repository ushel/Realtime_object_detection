import cv2
from ultralytics import YOLO
import supervision as sv

frame_width = 1280
frame_height = 720

def detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    
    model = YOLO("yolov8l.pt")
    
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale = 1
    )    
    
    while True:
        ret, frame = cap.read()
        
        results = model(frame,agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(results)
        
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _,  confidence, class_id, __
            in detections
        ]
        
        frame = box_annotator.annotate(
            scene = frame,
            detections= detections, 
            labels= labels
        )
        
        cv2.imshow('yolov8',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
if __name__ == "__main__":
    detection()