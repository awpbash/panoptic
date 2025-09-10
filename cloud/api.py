# cloud/api.py
import os
import time
import threading
from typing import Dict, Optional

import cv2
import numpy as np
import torch # Ensure torch is imported if RFDETRBase depends on it
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, Response # Import Response for image data
import base64 # For encoding image to base64

# Detection + tracking
from rfdetr import RFDETRBase
import supervision as sv

app = FastAPI(title="Cloud People Counter API")

# ---------------- Config (env) ----------------
# vertical counting line: fraction of image width (0.0 .. 1.0). 0.5 = center
LINE_X_FRAC = float(os.getenv("LINE_X_FRAC", "0.5"))
# default confidence threshold for detection
DEFAULT_CONF = float(os.getenv("CONF_THRES", "0.5"))
# Color for bounding boxes (BGR format for OpenCV)
BBOX_COLOR = (0, 255, 0) # Green
LINE_COLOR = (255, 0, 255) # Magenta
TEXT_COLOR = (255, 255, 255) # White

# ---------------- Lazy model ----------------
_model: Optional[RFDETRBase] = None
def get_model() -> RFDETRBase:
    global _model
    if _model is None:
        _model = RFDETRBase()
        _model.optimize_for_inference()
    return _model

# ---------------- Per-stream state ----------------
class StreamState:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.counts = {"left_to_right": 0, "right_to_left": 0}
        self.last_x: Dict[int, float] = {}
        self.counted_dir: Dict[int, str] = {}  # "L2R" or "R2L"
        self.last_update = 0.0
        self.lock = threading.Lock()

STREAMS: Dict[str, StreamState] = {}
def get_stream(stream_id: str) -> StreamState:
    if stream_id not in STREAMS:
        STREAMS[stream_id] = StreamState()
    return STREAMS[stream_id]

# ---------------- Helpers ----------------
def _count_crossings(st: StreamState, tracks_xyxy, track_ids, mid_x: float):
    """
    Update L->R / R->L counts if a tracked centroid crosses the vertical line mid_x.
    """
    for bbox, tid in zip(tracks_xyxy, track_ids):
        x1, y1, x2, y2 = bbox.astype(float)
        cx = 0.5 * (x1 + x2)
        last_x = st.last_x.get(int(tid), cx)
        st.last_x[int(tid)] = cx

        # left -> right crossing
        if last_x < mid_x <= cx:
            if st.counted_dir.get(int(tid)) != "L2R":
                st.counts["left_to_right"] += 1
                st.counted_dir[int(tid)] = "L2R"

        # right -> left crossing
        elif last_x > mid_x >= cx:
            if st.counted_dir.get(int(tid)) != "R2L":
                st.counts["right_to_left"] += 1
                st.counted_dir[int(tid)] = "R2L"

# ---------------- Endpoints ----------------
@app.post("/ingest")
async def ingest_frame(
    stream_id: str = Form(...),
    frame: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    draw_annotations: bool = Form(False, description="Whether to draw bounding boxes and line on the output image"),
    return_image: bool = Form(False, description="Whether to return the annotated image as base64 in the JSON response"),
):
    """
    Receive a single JPEG frame, run RF-DETR person detection, track with ByteTrack,
    and update L<->R counts (crossing a vertical line at LINE_X_FRAC).
    Optionally returns the annotated frame.
    """
    # Read and decode the frame
    data = await frame.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"ok": False, "error": "bad image"}, status_code=400)

    h, w = bgr.shape[:2]
    mid_x = float(w) * LINE_X_FRAC

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dets = get_model().predict(rgb, threshold=float(conf))
    people = dets[dets.class_id == 1]

    # Track
    st = get_stream(stream_id)
    tracks = st.tracker.update_with_detections(people)

    # Update counts
    with st.lock:
        _count_crossings(st, tracks.xyxy, tracks.tracker_id, mid_x)
        st.last_update = time.time()
        current_counts = st.counts.copy() # Get a copy for response

    # --- Annotation and Image Return ---
    encoded_img_str = None
    if draw_annotations or return_image:
        annotated_frame = bgr.copy() # Draw on a copy

        # Draw the counting line
        cv2.line(annotated_frame, (int(mid_x), 0), (int(mid_x), h), LINE_COLOR, 2)
        cv2.putText(annotated_frame, f"L2R: {current_counts['left_to_right']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINE_COLOR, 2)
        cv2.putText(annotated_frame, f"R2L: {current_counts['right_to_left']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINE_COLOR, 2)

        # Draw bounding boxes and IDs
        for bbox, tid in zip(tracks.xyxy, tracks.tracker_id):
            x1, y1, x2, y2 = [int(val) for val in bbox]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)
            # Display tracker ID
            label = f"ID: {tid}"
            cv2.putText(annotated_frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
        
        if return_image:
            # Encode the annotated frame to JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_img_str = base64.b64encode(buffer).decode('utf-8')

    # Respond with current counts and geometry info
    response_content = {
        "ok": True,
        "counts": current_counts,
        "w": w,
        "h": h,
        "mid_x": mid_x,
        "ts": st.last_update,
    }
    if return_image:
        response_content["annotated_image_b64"] = encoded_img_str
        
    return JSONResponse(response_content)

@app.get("/counts")
def get_counts(stream_id: str = Query(...)):
    st = get_stream(stream_id)
    with st.lock:
        return {"ok": True, "counts": st.counts, "ts": st.last_update}

@app.post("/reset")
def reset(stream_id: str = Form(...)):
    st = get_stream(stream_id)
    with st.lock:
        st.counts = {"left_to_right": 0, "right_to_left": 0}
        st.last_x.clear()
        st.counted_dir.clear()
        st.last_update = time.time()
        return {"ok": True, "counts": st.counts}

# (Optional) print routes at startup for sanity
@app.on_event("startup")
async def list_routes():
    print("Routes:")
    for r in app.routes:
        print(getattr(r, "methods", ""), r.path)