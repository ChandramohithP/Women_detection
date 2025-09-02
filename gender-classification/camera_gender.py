import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model + processor
MODEL_PATH = "D:/gd_model/gender-classification"  # full absolute path
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

# Gender labels (check config.json for label mapping if flipped)
labels = model.config.id2label

def classify_gender(img):
    # Convert BGR (OpenCV) -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()
    return labels[pred]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dummy detection: whole frame as "person" (replace this with YOLO for real detection)
    h, w, _ = frame.shape
    x, y, ww, hh = 50, 50, w-100, h-100
    person = frame[y:y+hh, x:x+ww]

    gender = classify_gender(person)

    color = (255, 0, 0) if gender.lower() == "male" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+ww, y+hh), color, 2)
    cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
