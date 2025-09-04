"""
realtime_person_gender.py

Production-style real-time person detection + gender classification + SOS gesture detection.

Features:
 - YOLOv8n (ultralytics) for fast person detection (GPU if available).
 - Fallback to OpenCV DNN YOLOv3-tiny if ultralytics unavailable.
 - IoU-based lightweight tracker with persistent IDs.
 - HuggingFace gender classifier with batch inference and CLAHE enhancement.
 - Voting buffer for stable gender labels.
 - Real-time Firebase updates:
     -> crowd_counts (male/female counts with timestamp)
     -> alerts (SOS gesture detected with timestamp)
 - Hand gesture SOS detection (thumb-in + open/close cycles).
 - Fullscreen OpenCV visualization with boxes, IDs, labels, counters, SOS banner.
"""

import time
import math
import sys
import traceback
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cvzone
from cvzone.HandTrackingModule import HandDetector

import firebase_admin
from firebase_admin import credentials, db

# ---------------- Firebase Init ----------------
cred = credentials.Certificate(
    r"D:\gd_model\gender-classification\Eagle-Eye\eagle_eye\firebase_key.json"
)
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://women-safety-fc31a-default-rtdb.asia-southeast1.firebasedatabase.app"
    },
)

# ---------------- CONFIG ----------------
USE_ULTRALYTICS = True
YOLOV8_NAME = "yolov8n.pt"
YOLO3_TINY_WEIGHTS = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\weights\yolov3-tiny.weights"
YOLO3_TINY_CFG = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\cfg\yolov3-tiny.cfg"
YOLO3_NAMES = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\lib\coco.names"

GENDER_MODEL_PATH = r"D:\gd_model\gender-classification"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTION_CONF_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.45
MIN_CROP_AREA = 1600
CLASSIFY_EVERY_N_FRAMES = 2
FEMALE_BIAS = 1.5
TRACK_MAX_AGE = 1.8
IOU_MATCH_THRESHOLD = 0.35
VOTE_BUFFER_SIZE = 7
STABLE_VOTE_MIN = 3
COUNTER_SMOOTH_WINDOW = 5
BATCH_MAX_CROPS = 12

WINDOW_NAME = "Realtime Person+Gender+SOS (press q to quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# SOS gesture params
thumb_closed = False
open_close_count = 0
last_finger_state = None
alert_triggered = False
alert_start_time = None
alert_duration = 3
sos_gesture_timeout = 5
required_open_close_count = 3
sos_pattern_start_time = None

# ---------------- Firebase helpers ----------------
def send_alert():
    ref = db.reference("alerts")
    ref.set({"status": "SOS DETECTED", "timestamp": str(time.time())})
    print("🚨 Alert sent to Firebase!")


def update_counts(males: int, females: int):
    ref = db.reference("crowd_counts")
    ref.set({"male": males, "female": females, "timestamp": str(time.time())})


# ---------------- Utility functions ----------------
def safe_rect(frame_w, frame_h, x, y, w, h):
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    w = int(max(1, min(w, frame_w - x1)))
    h = int(max(1, min(h, frame_h - y1)))
    return x1, y1, w, h


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def resize_preserve_aspect(img, target_short=224):
    h, w = img.shape[:2]
    if min(h, w) == 0:
        return None
    scale = target_short / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    cy, cx = nh // 2, nw // 2
    half = target_short // 2
    crop = resized[max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
    if crop.shape[0] != target_short or crop.shape[1] != target_short:
        crop = cv2.resize(crop, (target_short, target_short))
    return crop


def clahe_enhance(bgr_roi):
    try:
        lab = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except:
        return bgr_roi


# ---------------- Tracker ----------------
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: float
    history_votes: deque = field(default_factory=lambda: deque(maxlen=VOTE_BUFFER_SIZE))
    gender: Optional[str] = None
    gender_conf: float = 0.0
    stable: bool = False


class IoUTracker:
    def __init__(self, max_age=TRACK_MAX_AGE, iou_threshold=IOU_MATCH_THRESHOLD):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def step(self, detections: List[Tuple[int, int, int, int]]):
        now = time.time()
        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())
        iou_rows = []
        for tid, tr in self.tracks.items():
            iou_row = [iou(tr.bbox, d) for d in detections]
            iou_rows.append((tid, iou_row))
        used_dets = set()
        for tid, iou_row in sorted(
            iou_rows, key=lambda x: -max(x[1]) if x[1] else 0
        ):
            best_j, best_val = -1, 0
            for j, val in enumerate(iou_row):
                if val > best_val and j not in used_dets:
                    best_val, best_j = val, j
            if best_val >= self.iou_threshold and best_j >= 0:
                matches.append((tid, best_j))
                used_dets.add(best_j)
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(best_j)
        for tid, j in matches:
            self.tracks[tid].bbox = detections[j]
            self.tracks[tid].last_seen = now
        for j in list(unmatched_dets):
            self.tracks[self.next_id] = Track(self.next_id, detections[j], now)
            matches.append((self.next_id, j))
            self.next_id += 1
        for tid in [
            tid for tid, tr in self.tracks.items() if now - tr.last_seen > self.max_age
        ]:
            del self.tracks[tid]
        return matches, list(unmatched_dets)


# ---------------- Classifier ----------------
class GenderClassifier:
    def __init__(self, model_path, device=DEVICE, female_bias=FEMALE_BIAS):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_path
        ).to(device).eval()
        self.labels = self.model.config.id2label
        self.female_bias = female_bias
        print(f"[INFO] Gender model loaded: {self.labels}")

    def predict_batch(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        if len(crops) == 0:
            return []
        processed, valid_idx = [], []
        for i, c in enumerate(crops):
            if c.shape[0] * c.shape[1] < MIN_CROP_AREA:
                processed.append(None)
                continue
            square = resize_preserve_aspect(clahe_enhance(c), 224)
            processed.append(square)
            valid_idx.append(i) if square is not None else None
        if not valid_idx:
            return [("unknown", 0.0)] * len(crops)
        inputs = self.processor(
            images=[processed[i] for i in valid_idx], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            probs = (
                torch.nn.functional.softmax(
                    self.model(**inputs).logits, dim=-1
                )
                .cpu()
                .numpy()
            )
        female_idx = next(
            (k for k, v in self.labels.items() if v.lower() == "female"), None
        )
        if female_idx is not None:
            probs[:, female_idx] *= self.female_bias
        probs /= probs.sum(axis=1, keepdims=True)
        results_valid = [
            (self.labels[int(np.argmax(p))], float(np.max(p))) for p in probs
        ]
        results, vi = [], 0
        for i in range(len(crops)):
            if i in valid_idx:
                results.append(results_valid[vi])
                vi += 1
            else:
                results.append(("unknown", 0.0))
        return results


# ---------------- Main Loop ----------------
def main_loop(video_src=0):
    # Detector init
    detector = None
    try:
        if USE_ULTRALYTICS:
            from ultralytics import YOLO

            detector = {"type": "ultralytics", "model": YOLO(YOLOV8_NAME)}
            print("[INFO] YOLOv8 loaded.")
        else:
            net = cv2.dnn.readNet(YOLO3_TINY_WEIGHTS, YOLO3_TINY_CFG)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with open(YOLO3_NAMES) as f:
                names = [l.strip() for l in f]
            detector = {"type": "cv2-dnn", "net": net, "names": names}
    except Exception as e:
        print("[FATAL] Detector failed:", e)
        sys.exit(1)

    # Classifier & Tracker
    classifier = GenderClassifier(GENDER_MODEL_PATH)
    tracker = IoUTracker()

    # Hand detector
    hand_detector = HandDetector(maxHands=2)

    cap = cv2.VideoCapture(video_src)
    cap.set(3, 640)
    cap.set(4, 480)
    if not cap.isOpened():
        print("[ERROR] Camera failed.")
        return

    counter_buffer = deque(maxlen=COUNTER_SMOOTH_WINDOW)
    frame_id, last_classify_frame = 0, -999
    start_time = time.time()

    global thumb_closed, open_close_count, last_finger_state, alert_triggered, alert_start_time, sos_pattern_start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        H, W = frame.shape[:2]

        # 1. Person detection
        detections = []
        if detector["type"] == "ultralytics":
            res = detector["model"].predict(
                source=frame, imgsz=640, conf=DETECTION_CONF_THRESHOLD, verbose=False
            )[0]
            for b, c, cl in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
            ):
                if cl != 0:
                    continue
                x1, y1, x2, y2 = map(int, b.tolist())
                detections.append([x1, y1, x2 - x1, y2 - y1, float(c)])
        else:
            net = detector["net"]
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
            )
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())
            for out in outs:
                for det in out:
                    scores = det[5:]
                    class_id = int(np.argmax(scores))
                    conf = float(scores[class_id])
                    if class_id == 0 and conf >= DETECTION_CONF_THRESHOLD:
                        cx, cy, w, h = (
                            int(det[0] * W),
                            int(det[1] * H),
                            int(det[2] * W),
                            int(det[3] * H),
                        )
                        detections.append([cx - w // 2, cy - h // 2, w, h, conf])

        kept_boxes = detections

        # Tracker
        matches, _ = tracker.step([[b[0], b[1], b[2], b[3]] for b in kept_boxes])

        # Classification
        classify_now = (frame_id - last_classify_frame) >= CLASSIFY_EVERY_N_FRAMES
        if classify_now:
            last_classify_frame = frame_id
        crops, ids = [], []
        for tid, det_idx in matches:
            x, y, w, h = safe_rect(W, H, *kept_boxes[det_idx][:4])
            if w * h < MIN_CROP_AREA:
                continue
            if classify_now:
                crops.append(frame[y : y + h, x : x + w])
                ids.append(tid)
        results = classifier.predict_batch(crops) if crops else []
        for tid, (label, conf) in zip(ids, results):
            tr = tracker.tracks.get(tid)
            tr.history_votes.append(label)
            tr.gender_conf = conf
            cnt = Counter(tr.history_votes)
            if cnt:
                maj, mcnt = cnt.most_common(1)[0]
                tr.gender = maj
                tr.stable = mcnt >= STABLE_VOTE_MIN

        # Counters
        male_total = female_total = 0
        display = frame.copy()
        for tid, tr in tracker.tracks.items():
            x, y, w, h = safe_rect(W, H, *tr.bbox)
            if tr.gender:
                g = tr.gender.lower()
                if "female" in g:
                    female_total += 1 if tr.stable else 0
                    color = (200, 50, 200)
                elif "male" in g:
                    male_total += 1 if tr.stable else 0
                    color = (50, 150, 255)
                else:
                    color = (180, 180, 180)
                label = f"ID{tid} {tr.gender[:1]} {tr.gender_conf:.2f}"
            else:
                label = f"ID{tid} ?"
                color = (180, 180, 180)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2 if tr.stable else 1)
            cv2.putText(display, label, (x, max(12, y - 6)), FONT, 0.5, color, 2)

        counter_buffer.append((male_total, female_total))
        m_avg = int(np.mean([c[0] for c in counter_buffer])) if counter_buffer else 0
        f_avg = int(np.mean([c[1] for c in counter_buffer])) if counter_buffer else 0

        update_counts(m_avg, f_avg)  # Firebase counts

        cv2.rectangle(display, (0, 0), (320, 80), (0, 0, 0), -1)
        cv2.putText(display, f"👨 Males: {m_avg}", (10, 28), FONT, 0.7, (0, 200, 255), 2)
        cv2.putText(
            display, f"👩 Females: {f_avg}", (10, 60), FONT, 0.7, (200, 30, 200), 2
        )

        # 2. Hand gesture SOS detection
        hands, img_h = hand_detector.findHands(frame.copy())
        for hand in hands:
            fingers = hand_detector.fingersUp(hand)
            if fingers[0] == 0 and not thumb_closed:
                thumb_closed = True
                open_close_count = 0
                last_finger_state = None
                sos_pattern_start_time = time.time()
                print("Thumb closed → begin SOS sequence.")
            if thumb_closed and sos_pattern_start_time:
                elapsed = time.time() - sos_pattern_start_time
                if elapsed > sos_gesture_timeout:
                    thumb_closed = False
                    sos_pattern_start_time = None
                    break
                state = (
                    "open"
                    if fingers[1:] == [1, 1, 1, 1]
                    else "closed"
                    if fingers[1:] == [0, 0, 0, 0]
                    else None
                )
                if state and state != last_finger_state:
                    if state == "closed" and last_finger_state == "open":
                        open_close_count += 1
                        print(
                            f"Open-close: {open_close_count}/{required_open_close_count}"
                        )
                    last_finger_state = state
                if open_close_count >= required_open_close_count:
                    alert_triggered = True
                    alert_start_time = time.time()
                    send_alert()
                    print("🚨 SOS Detected!")
        if alert_triggered and (time.time() - alert_start_time < alert_duration):
            cv2.putText(display, "🚨 SOS DETECTED", (50, 120), FONT, 2, (0, 0, 255), 4)
        elif alert_triggered:
            alert_triggered = False

        # FPS
        fps = frame_id / (time.time() - start_time)
        cv2.putText(display, f"FPS:{fps:.1f}", (W - 120, 28), FONT, 0.6, (200, 200, 0), 2)

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- Entry Point ----------------
if __name__ == "__main__":
    try:
        main_loop(0)  # use default webcam
    except Exception as e:
        print("[FATAL] Unhandled exception:", e)
        traceback.print_exc()
