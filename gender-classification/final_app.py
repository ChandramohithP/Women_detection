"""
Safety Monitor – Unified Person/Gender + SOS Gesture (Python)
----------------------------------------------------------------
A production-style, modular sample that combines:
  • Person detection + tracking + gender classification (YOLOv8 or YOLOv3-tiny fallback)
  • SOS hand-gesture detection (cvzone.HandTrackingModule)
  • Firebase alerting (instant, robust with retries)
  • Fullscreen UI overlay with counts, IDs, gender, and SOS banner
  • Optional incident recording (pre/post buffer)

Design goals:
  • Single camera stream → shared preprocessing → parallel pipelines → unified compositor
  • Thread-safe, low-latency (queues with maxsize=1 to avoid backlog)
  • Robust error handling; safe shutdown
  • Easily deployable on edge devices (Jetson/RPi/mini-PC)

NOTE: Fill in CONFIG paths for your environment (Firebase key & URL, model paths).
"""

from __future__ import annotations
import os
import sys
import time
import math
import traceback
import threading
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# ---- Optional / soft deps: wrap imports so the file still runs where some libs are missing ----
try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO  # for YOLOv8
except Exception:
    YOLO = None

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification  # HF gender model
except Exception:
    AutoImageProcessor = AutoModelForImageClassification = None

try:
    import cvzone
    from cvzone.HandTrackingModule import HandDetector
except Exception:
    HandDetector = None

try:
    import firebase_admin
    from firebase_admin import credentials, db
except Exception:
    firebase_admin = credentials = db = None

# =====================================
# CONFIG
# =====================================
class CONFIG:
    # --- Camera ---
    VIDEO_SRC = 0  # integer index, RTSP/URL, or file
    CAM_WIDTH = 640
    CAM_HEIGHT = 480

    # --- Detector choice ---
    USE_ULTRALYTICS = True               # try YOLOv8; falls back to cv2.dnn if unavailable
    YOLOV8_NAME = "yolov8n.pt"           # ultralytics auto-downloads if missing

    # Fallback (cv2.dnn YOLOv3-tiny)
    YOLO3_TINY_WEIGHTS = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\weights\yolov3-tiny.weights"
    YOLO3_TINY_CFG     = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\cfg\yolov3-tiny.cfg"
    YOLO3_NAMES        = r"D:\gd_model\gender-classification\persondetection\YOLO-Realtime-Human-Detection\lib\coco.names"

    # --- Gender classifier (HuggingFace local dir with config/model)—matches your existing model folder
    GENDER_MODEL_PATH = r"D:\gd_model\gender-classification"

    # --- Firebase ---
    FIREBASE_KEY_PATH = r"D:\gd_model\gender-classification\Eagle-Eye\eagle_eye\firebase_key.json"
    FIREBASE_DB_URL   = "https://women-safety-fc31a-default-rtdb.asia-southeast1.firebasedatabase.app"
    FIREBASE_ALERT_NODE = "alerts"

    # --- Runtime thresholds ---
    DEVICE                = "cuda" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    DETECTION_CONF        = 0.35
    NMS_IOU               = 0.45
    MIN_CROP_AREA         = 1600
    CLASSIFY_EVERY_N      = 2
    FEMALE_BIAS           = 1.5
    TRACK_MAX_AGE_SEC     = 1.8
    IOU_MATCH_THR         = 0.35
    VOTE_BUFFER           = 7
    STABLE_VOTE_MIN       = 3
    COUNTER_SMOOTH_WIN    = 5
    BATCH_MAX_CROPS       = 12

    # --- UI ---
    WINDOW_NAME = "Safety Monitor (q=quit)"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # --- Low-light / IR handling ---
    ENABLE_LOW_LIGHT_ENHANCE = True
    LOW_LIGHT_MEAN_THR = 60      # mean pixel threshold to trigger low-light pipeline

    # --- SOS gesture detection ---
    SOS_REQUIRED_CYCLES = 3
    SOS_TIMEOUT_SEC = 5
    SOS_BANNER_HOLD_SEC = 3

    # --- Incident recording ---
    RECORD_ON_ALERT = True
    PRE_BUFFER_SEC = 5
    POST_BUFFER_SEC = 10
    RECORD_FPS = 20
    RECORD_DIR = "recordings"

    # --- Misc ---
    VERBOSE = False


def vlog(*args):
    if CONFIG.VERBOSE:
        print(*args)

# =====================================
# Shared preprocessing (CLAHE + aspect-safe resize utilities)
# =====================================

def clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr


def detect_low_light(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < CONFIG.LOW_LIGHT_MEAN_THR


def safe_rect(frame_w, frame_h, x, y, w, h):
    x1 = max(0, int(x)); y1 = max(0, int(y))
    w = int(max(1, min(w, frame_w - x1)))
    h = int(max(1, min(h, frame_h - y1)))
    return x1, y1, w, h


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]; boxBArea = boxB[2] * boxB[3]
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def resize_preserve_aspect(img: np.ndarray, target_short=224) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    if min(h, w) == 0:
        return None
    scale = target_short / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    cy, cx = nh // 2, nw // 2
    half = target_short // 2
    y1 = max(0, cy - half); x1 = max(0, cx - half)
    crop = resized[y1:y1 + target_short, x1:x1 + target_short]
    if crop.shape[0] != target_short or crop.shape[1] != target_short:
        crop = cv2.resize(crop, (target_short, target_short))
    return crop

# =====================================
# IoU Tracker (lightweight)
# =====================================
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: float
    votes: deque = field(default_factory=lambda: deque(maxlen=CONFIG.VOTE_BUFFER))
    gender: Optional[str] = None
    conf: float = 0.0
    stable: bool = False


class IoUTracker:
    def __init__(self, max_age=CONFIG.TRACK_MAX_AGE_SEC, thr=CONFIG.IOU_MATCH_THR):
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.max_age = max_age
        self.thr = thr

    def step(self, detections: List[Tuple[int,int,int,int]]):
        now = time.time()
        matches = []
        unmatched = set(range(len(detections)))
        # build iou rows
        rows = []
        for tid, tr in self.tracks.items():
            rows.append((tid, [iou(tr.bbox, d) for d in detections]))
        # greedy match
        used = set()
        for tid, row in sorted(rows, key=lambda x: -max(x[1]) if x[1] else 0):
            best_j, best_v = -1, 0
            for j, v in enumerate(row):
                if j not in used and v > best_v:
                    best_j, best_v = j, v
            if best_v >= self.thr and best_j >= 0:
                matches.append((tid, best_j))
                used.add(best_j)
                if best_j in unmatched: unmatched.remove(best_j)
        # update/create
        for tid, j in matches:
            self.tracks[tid].bbox = detections[j]
            self.tracks[tid].last_seen = now
        for j in list(unmatched):
            self.tracks[self.next_id] = Track(self.next_id, detections[j], now)
            matches.append((self.next_id, j))
            self.next_id += 1
        # prune
        stale = [tid for tid, tr in self.tracks.items() if now - tr.last_seen > self.max_age]
        for tid in stale:
            del self.tracks[tid]
        return matches, list(unmatched)

# =====================================
# Detector + Classifier (Person/Gender)
# =====================================
class GenderClassifier:
    def __init__(self, model_path: str, device: str, female_bias: float):
        if AutoImageProcessor is None or AutoModelForImageClassification is None:
            raise RuntimeError("transformers not available")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.labels = self.model.config.id2label
        self.female_bias = female_bias
        print(f"[INFO] Gender model loaded: {model_path}; device={device}; labels={self.labels}")

    def predict_batch(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not crops:
            return []
        processed = []
        for c in crops:
            h, w = c.shape[:2]
            if h * w < CONFIG.MIN_CROP_AREA:
                processed.append(None)
                continue
            c2 = clahe_enhance(c)
            s = resize_preserve_aspect(c2, 224)
            processed.append(s)
        valid_idx = [i for i, p in enumerate(processed) if p is not None]
        if not valid_idx:
            return [("unknown", 0.0) for _ in crops]
        images = [processed[i] for i in valid_idx]
        inputs = self.processor(images=images, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        # female bias
        female_idx = next((k for k, v in self.labels.items() if v.lower() == "female"), None)
        if female_idx is not None:
            probs[:, female_idx] *= CONFIG.FEMALE_BIAS
            probs = probs / probs.sum(axis=1, keepdims=True)
        res_valid = []
        for p in probs:
            idx = int(np.argmax(p))
            res_valid.append((self.labels[idx], float(p[idx])))
        # map back
        results = []
        vi = 0
        for i in range(len(crops)):
            results.append(res_valid[vi] if i in valid_idx else ("unknown", 0.0))
            if i in valid_idx: vi += 1
        return results


class PersonDetector:
    def __init__(self):
        self.mode = "ultralytics" if CONFIG.USE_ULTRALYTICS and YOLO is not None else "cv2-dnn"
        if self.mode == "ultralytics":
            try:
                self.model = YOLO(CONFIG.YOLOV8_NAME)
                print("[INFO] Using YOLOv8:", CONFIG.YOLOV8_NAME)
            except Exception as e:
                print("[WARN] YOLOv8 failed → fallback cv2.dnn:", e)
                self.mode = "cv2-dnn"
        if self.mode == "cv2-dnn":
            net = cv2.dnn.readNet(CONFIG.YOLO3_TINY_WEIGHTS, CONFIG.YOLO3_TINY_CFG)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with open(CONFIG.YOLO3_NAMES, "r") as f:
                names = [l.strip() for l in f]
            self.net = net
            self.names = names
            print("[INFO] Using YOLOv3-tiny (cv2.dnn) fallback")

    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        H, W = frame.shape[:2]
        dets = []  # x,y,w,h,conf
        if self.mode == "ultralytics":
            try:
                res = self.model.predict(source=frame, imgsz=640, conf=CONFIG.DETECTION_CONF, verbose=False)[0]
                boxes = res.boxes
                if boxes is None: return dets
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
                confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
                clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []
                for b, c, cl in zip(xyxy, confs, clss):
                    if cl != 0:  # class 0 = person
                        continue
                    x1, y1, x2, y2 = map(int, b.tolist())
                    dets.append([x1, y1, x2 - x1, y2 - y1, float(c)])
            except Exception as e:
                vlog("[WARN] YOLOv8 predict error:", e)
        else:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            try:
                outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            except Exception:
                outs = self.net.forward()
            for out in outs:
                for det in out:
                    scores = det[5:]
                    if scores.size == 0: continue
                    cid = int(np.argmax(scores)); conf = float(scores[cid])
                    if cid == 0 and conf >= CONFIG.DETECTION_CONF:
                        cx = int(det[0] * W); cy = int(det[1] * H)
                        w = int(det[2] * W); h = int(det[3] * H)
                        x = cx - w // 2; y = cy - h // 2
                        dets.append([int(x), int(y), int(w), int(h), conf])
        # NMS (OpenCV)
        if not dets:
            return dets
        boxes_np = np.array([d[:4] for d in dets])
        scores_np = np.array([d[4] for d in dets])
        x1 = boxes_np[:, 0]; y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 0] + boxes_np[:, 2]; y2 = boxes_np[:, 1] + boxes_np[:, 3]
        x1 = np.clip(x1, 0, W-1); y1 = np.clip(y1, 0, H-1)
        x2 = np.clip(x2, 0, W-1); y2 = np.clip(y2, 0, H-1)
        rects = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
        try:
            idxs = cv2.dnn.NMSBoxes(rects.tolist(), scores_np.tolist(), CONFIG.DETECTION_CONF, CONFIG.NMS_IOU)
            if isinstance(idxs, np.ndarray):
                idxs = idxs.flatten().tolist()
            elif isinstance(idxs, (list, tuple)) and idxs and isinstance(idxs[0], (list, tuple, np.ndarray)):
                idxs = [int(i[0]) for i in idxs]
        except Exception:
            # naive NMS
            order = np.argsort(-scores_np)
            keep = []
            for i in order:
                if all(iou(boxes_np[i], boxes_np[k]) <= CONFIG.NMS_IOU for k in keep):
                    keep.append(i)
            idxs = keep
        return [dets[i] for i in idxs]


# =====================================
# SOS Gesture Engine (cvzone)
# =====================================
class SOSGesture:
    def __init__(self):
        if HandDetector is None:
            raise RuntimeError("cvzone HandDetector not available")
        self.detector = HandDetector(maxHands=2)
        self.thumb_closed = False
        self.open_close_count = 0
        self.last_state = None
        self.pattern_start = None
        self.alert_active_until = 0.0

    def update(self, frame: np.ndarray) -> bool:
        """Process one frame; return True if SOS reliably detected.
        Also draws minimal helper overlays on the given frame.
        """
        hands, _ = self.detector.findHands(frame, draw=False)
        now = time.time()
        sos_detected = False
        for hand in hands:
            fingers = self.detector.fingersUp(hand)
            # start when thumb goes closed
            if fingers[0] == 0 and not self.thumb_closed:
                self.thumb_closed = True
                self.open_close_count = 0
                self.last_state = None
                self.pattern_start = now
            if self.thumb_closed and self.pattern_start:
                if now - self.pattern_start > CONFIG.SOS_TIMEOUT_SEC:
                    # reset on timeout
                    self.thumb_closed = False
                    self.pattern_start = None
                    break
                current = "open" if fingers[1:] == [1,1,1,1] else ("closed" if fingers[1:] == [0,0,0,0] else None)
                if current and current != self.last_state:
                    if current == "closed" and self.last_state == "open":
                        self.open_close_count += 1
                    self.last_state = current
                if self.open_close_count >= CONFIG.SOS_REQUIRED_CYCLES:
                    sos_detected = True
                    self.alert_active_until = now + CONFIG.SOS_BANNER_HOLD_SEC
                    # reset for next sequence
                    self.thumb_closed = False
                    self.pattern_start = None
                    break
        # visual hint (small corner text)
        if now < self.alert_active_until:
            cv2.putText(frame, "SOS DETECTED", (10, 30), CONFIG.FONT, 0.8, (0,0,255), 2)
        return sos_detected


# =====================================
# Firebase client with retries
# =====================================
class FirebaseClient:
    def __init__(self):
        if firebase_admin is None:
            print("[WARN] firebase_admin not installed; alerts disabled")
            self.ok = False
            return
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(CONFIG.FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {"databaseURL": CONFIG.FIREBASE_DB_URL})
            self.ref = db.reference(CONFIG.FIREBASE_ALERT_NODE)
            self.ok = True
            print("[INFO] Firebase initialized")
        except Exception as e:
            print("[ERROR] Firebase init failed:", e)
            self.ok = False

    def send_alert(self):
        if not self.ok:
            return False
        payload = {"status": "SOS DETECTED", "timestamp": str(time.time())}
        for attempt in range(3):
            try:
                self.ref.set(payload)
                print("🚨 Firebase alert sent")
                return True
            except Exception as e:
                print(f"[WARN] Firebase send attempt {attempt+1} failed: {e}")
                time.sleep(0.5)
        return False


# =====================================
# Incident recorder (ring buffer pre/post)
# =====================================
class IncidentRecorder:
    def __init__(self, fps: int, width: int, height: int):
        self.enabled = CONFIG.RECORD_ON_ALERT
        self.buf = deque(maxlen=int(CONFIG.PRE_BUFFER_SEC * fps))
        self.post_frames_left = 0
        self.writer = None
        self.fps = fps
        self.size = (width, height)
        os.makedirs(CONFIG.RECORD_DIR, exist_ok=True)

    def add_frame(self, frame: np.ndarray):
        if not self.enabled:
            return
        self.buf.append(frame.copy())
        if self.post_frames_left > 0 and self.writer is not None:
            self.writer.write(frame)
            self.post_frames_left -= 1
            if self.post_frames_left == 0:
                self._close()

    def trigger(self):
        if not self.enabled:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CONFIG.RECORD_DIR, f"incident_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, self.size)
        # dump pre-buffer
        for f in list(self.buf):
            self.writer.write(f)
        self.post_frames_left = int(CONFIG.POST_BUFFER_SEC * self.fps)
        print(f"[INFO] Recording incident → {path}")

    def _close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("[INFO] Incident recording saved")


# =====================================
# Worker Threads
# =====================================
class FrameBroadcaster(threading.Thread):
    """Captures frames once and makes the latest frame available to all consumers.
    Consumers call get_latest() which returns a copy without blocking the capture loop.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(CONFIG.VIDEO_SRC)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {CONFIG.VIDEO_SRC}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.CAM_HEIGHT)
        self.lock = threading.Lock()
        self.latest = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Camera read failed; retrying...")
                time.sleep(0.02)
                continue
            with self.lock:
                self.latest = frame
        self.cap.release()

    def get_latest(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False


class PersonGenderWorker(threading.Thread):
    def __init__(self, broadcaster: FrameBroadcaster):
        super().__init__(daemon=True)
        self.bc = broadcaster
        self.detector = PersonDetector()
        if torch is None:
            raise RuntimeError("PyTorch required for gender classifier")
        self.classifier = GenderClassifier(CONFIG.GENDER_MODEL_PATH, CONFIG.DEVICE, CONFIG.FEMALE_BIAS)
        self.tracker = IoUTracker()
        self.counter_buf = deque(maxlen=CONFIG.COUNTER_SMOOTH_WIN)
        self.overlay = None
        self.m_avg = 0
        self.f_avg = 0
        self.running = True
        self.frame_id = 0
        self.last_cls_frame = -999

    def run(self):
        while self.running:
            frame = self.bc.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue
            self.frame_id += 1
            H, W = frame.shape[:2]

            # Shared low-light handling
            if CONFIG.ENABLE_LOW_LIGHT_ENHANCE and detect_low_light(frame):
                proc = clahe_enhance(frame)
            else:
                proc = frame

            dets = self.detector.detect_persons(proc)
            boxes_only = [[d[0], d[1], d[2], d[3]] for d in dets]
            matches, _ = self.tracker.step(boxes_only)

            classify_now = (self.frame_id - self.last_cls_frame) >= CONFIG.CLASSIFY_EVERY_N
            if classify_now:
                self.last_cls_frame = self.frame_id

            crops, ids = [], []
            for (tid, j) in matches:
                x, y, w, h = safe_rect(W, H, *boxes_only[j])
                if w * h < CONFIG.MIN_CROP_AREA:  # skip tiny
                    continue
                if classify_now and len(crops) < CONFIG.BATCH_MAX_CROPS:
                    crops.append(proc[y:y+h, x:x+w].copy())
                    ids.append(tid)
            results = self.classifier.predict_batch(crops) if crops else []
            for tid, (label, conf) in zip(ids, results):
                tr = self.tracker.tracks.get(tid)
                if not tr: continue
                tr.votes.append(label)
                tr.conf = conf
                if tr.votes:
                    cnt = Counter(tr.votes)
                    maj, c = cnt.most_common(1)[0]
                    tr.gender = maj
                    tr.stable = c >= CONFIG.STABLE_VOTE_MIN

            # Draw overlay
            disp = frame.copy()
            male_total, female_total = 0, 0
            for tid, tr in self.tracker.tracks.items():
                x, y, w, h = tr.bbox
                x, y, w, h = safe_rect(W, H, x, y, w, h)
                label = f"ID{tid}"
                color = (180,180,180)
                if tr.gender:
                    g = tr.gender.lower()
                    if "female" in g or "woman" in g:
                        color = (200,50,200)
                        if tr.stable: female_total += 1
                        label += f" F {tr.conf:.2f}"
                    elif "male" in g or "man" in g:
                        color = (50,150,255)
                        if tr.stable: male_total += 1
                        label += f" M {tr.conf:.2f}"
                    else:
                        label += " ?"
                thick = 2 if tr.stable else 1
                cv2.rectangle(disp, (x,y), (x+w, y+h), color, thick)
                cv2.putText(disp, label, (x, max(12, y-6)), CONFIG.FONT, 0.5, color, 2)
            self.counter_buf.append((male_total, female_total))
            self.m_avg = int(np.mean([c[0] for c in self.counter_buf])) if self.counter_buf else 0
            self.f_avg = int(np.mean([c[1] for c in self.counter_buf])) if self.counter_buf else 0

            # Counter panel
            cv2.rectangle(disp, (0,0), (300,80), (0,0,0), -1)
            cv2.putText(disp, f"👨 Males (stable): {self.m_avg}", (10,28), CONFIG.FONT, 0.7, (0,200,255), 2)
            cv2.putText(disp, f"👩 Females (stable): {self.f_avg}", (10,60), CONFIG.FONT, 0.7, (200,30,200), 2)
            self.overlay = disp

    def stop(self):
        self.running = False


class SOSWorker(threading.Thread):
    def __init__(self, broadcaster: FrameBroadcaster, fb: FirebaseClient, recorder: IncidentRecorder):
        super().__init__(daemon=True)
        self.bc = broadcaster
        self.engine = SOSGesture()
        self.fb = fb
        self.rec = recorder
        self.running = True
        self.last_alert_ts = 0.0

    def run(self):
        while self.running:
            frame = self.bc.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue
            # Share same preprocessing heuristic as the person pipeline
            if CONFIG.ENABLE_LOW_LIGHT_ENHANCE and detect_low_light(frame):
                proc = clahe_enhance(frame)
            else:
                proc = frame
            detected = self.engine.update(proc)
            # continuously feed frames to recorder buffer
            self.rec.add_frame(proc)
            if detected and time.time() - self.last_alert_ts > 1.0:  # debounce
                self.last_alert_ts = time.time()
                # Fire alert (non-blocking feel; call in thread)
                threading.Thread(target=self.fb.send_alert, daemon=True).start()
                self.rec.trigger()

    def stop(self):
        self.running = False


# =====================================
# Main – compositor and control
# =====================================

def main():
    print("[INFO] Starting Safety Monitor ...")
    # Init
    fb = FirebaseClient()

    # Open a temp capture to query frame size for recorder in case broadcaster is delayed
    cap_probe = cv2.VideoCapture(CONFIG.VIDEO_SRC)
    cap_probe.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.CAM_WIDTH)
    cap_probe.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.CAM_HEIGHT)
    ret, frm0 = cap_probe.read()
    if not ret:
        raise RuntimeError("Camera probe failed")
    H0, W0 = frm0.shape[:2]
    cap_probe.release()

    recorder = IncidentRecorder(CONFIG.RECORD_FPS, W0, H0)

    broadcaster = FrameBroadcaster()
    broadcaster.start()

    person_worker = PersonGenderWorker(broadcaster)
    person_worker.start()

    sos_worker = SOSWorker(broadcaster, fb, recorder)
    sos_worker.start()

    start_time = time.time()

    try:
        while True:
            base = broadcaster.get_latest()
            if base is None:
                time.sleep(0.01)
                continue
            # Compose: prefer person overlay (already includes counters). Also show global FPS.
            disp = person_worker.overlay if person_worker.overlay is not None else base

            # If SOS banner was set inside SOSWorker engine, we want to display it prominently too:
            # The engine draws small text on its own frame, but here we add a big banner for clarity.
            # Simple heuristic: show banner if last alert was in last SOS_BANNER_HOLD_SEC.
            if time.time() - sos_worker.last_alert_ts < CONFIG.SOS_BANNER_HOLD_SEC:
                H, W = disp.shape[:2]
                cv2.rectangle(disp, (0, 0), (W, 60), (0, 0, 255), -1)
                cv2.putText(disp, "🚨 SOS DETECTED – ALERT SENT", (10, 40), CONFIG.FONT, 1.0, (255, 255, 255), 3)

            # FPS
            elapsed = max(1e-3, time.time() - start_time)
            fps = int((cv2.getTickFrequency() / max(1, cv2.getTickCount())) or 0)  # placeholder
            cv2.putText(disp, f"Uptime: {int(elapsed)}s", (10, disp.shape[0]-10), CONFIG.FONT, 0.6, (255,255,0), 2)

            cv2.namedWindow(CONFIG.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(CONFIG.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(CONFIG.WINDOW_NAME, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested")
                break
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt – exiting")
    except Exception as e:
        print("[ERROR] Main loop error:", e)
        traceback.print_exc()
    finally:
        person_worker.stop(); sos_worker.stop(); broadcaster.stop()
        person_worker.join(timeout=1.0)
        sos_worker.join(timeout=1.0)
        broadcaster.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
