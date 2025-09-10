import os
import time
import cv2
import numpy as np
import gradio as gr
import torch
import supervision as sv

# Your RF-DETR wrapper (Roboflow) — must be installed/available
from rfdetr import RFDETRBase
# (If you need class names: from rfdetr.util.coco_classes import COCO_CLASSES)

# ------------------ Globals ------------------
detr_model = None
tracker = None
device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_VIDEO_PATH = "../faster.mp4" # change if you want
# ------------------ Model load ------------------
def load_detr_model():
    """Loads RF-DETR and ByteTrack exactly once."""
    global detr_model, tracker
    if detr_model is not None and tracker is not None:
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return f"Model already loaded on {dev_name}."

    try:
        detr_model = RFDETRBase()
        detr_model.optimize_for_inference()
        tracker = sv.ByteTrack()
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return f"RF-DETR + ByteTrack loaded on {dev_name}."
    except Exception as e:
        return f"Error loading model: {e}"

# ------------------ Helpers ------------------
def _normalize_video_path(v):
    """Gradio Video can return a str path or dict with 'name' (tmp path)."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # Newer gradio returns {"name": "/tmp/....mp4", "orig_name": "..."}
        return v.get("name") or v.get("path") or v.get("video")
    return None

def get_first_frame(video_val):
    """
    Loads the first frame of the video and initializes state variables.
    
    Returns:
      - frame_rgb (numpy, RGB) for display,
      - normalized path (for state),
      - same frame as base image (state),
      - status message,
      - empty list for ROI points (state)
    """
    video_path = _normalize_video_path(video_val) or DEFAULT_VIDEO_PATH
    if not os.path.exists(video_path):
        return None, None, None, "Video not found.", []

    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None, None, None, "Failed to read first frame.", []

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    msg = "Frame loaded. Click to add ROI points (min 3)."
    # Return all states and display components in one go
    return frame_rgb, video_path, frame_rgb, msg, []

def _draw_overlay(base_img_rgb, pts_xy):
    """Draws the clicked points and open polyline on a copy of the base image."""
    vis = base_img_rgb.copy()
    # draw lines (open until 'Finish ROI')
    if len(pts_xy) >= 2:
        for a, b in zip(pts_xy[:-1], pts_xy[1:]):
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 2)
    # draw points + indices
    for i, (x, y) in enumerate(pts_xy):
        cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), -1)  # red dots
        cv2.putText(vis, str(i + 1), (int(x) + 5, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return vis

def handle_click(base_img, pts, evt: gr.SelectData):
    """
    Add a point where the user clicked.
    evt.index gives (x_disp, y_disp) in displayed coords, which are
    the same as original coords since the image is handled by Gradio.
    """
    if base_img is None:
        return None, pts, "Load a frame first."
    if pts is None:
        pts = []

    x, y = evt.index
    pts2 = pts + [(x, y)]
    vis = _draw_overlay(base_img, pts2)
    return vis, pts2, f"Points: {len(pts2)}"

def undo_point(base_img, pts):
    if base_img is None:
        return None, pts, "Load a frame first."
    if not pts:
        return base_img, pts, "No points to undo."
    pts2 = pts[:-1]
    vis = _draw_overlay(base_img, pts2)
    return vis, pts2, f"Points: {len(pts2)}"

def clear_points(base_img):
    if base_img is None:
        return None, [], "Load a frame first."
    vis = _draw_overlay(base_img, [])
    return vis, [], "Cleared points."

def finish_roi(base_img, pts):
    """Close polygon visually."""
    if base_img is None:
        return base_img, pts, "Load a frame first."
    if len(pts) < 3:
        return base_img, pts, "Need at least 3 points."
    
    # Create a copy of the base image to draw on
    vis = base_img.copy()
    
    # Draw existing points
    vis = _draw_overlay(vis, pts)
    
    # Draw the final closed polygon
    poly = np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)
    cv2.polylines(vis, [poly], True, (0, 255, 0), 2)
    
    return vis, pts, f"ROI set with {len(pts)} points."

# ------------------ Processing ------------------
def process_and_display_video(video_path_state, roi_points_state):
    """
    Runs detection+tracking; measures dwell inside the polygon; flags loitering.
    Returns: processed video path, list of RGB crops, timestamp text, metrics markdown.
    """
    if detr_model is None or tracker is None:
        return "Load the detection model first.", None, None, "Metrics not available."

    video_path = _normalize_video_path(video_path_state) or DEFAULT_VIDEO_PATH
    if not os.path.exists(video_path):
        return "Error: Video not found.", None, None, "Metrics not available."

    roi_poly = None
    if roi_points_state and len(roi_points_state) >= 3:
        roi_poly = np.array([[int(x), int(y)] for x, y in roi_points_state], dtype=np.int32)

    start = time.time()
    loiter_data = {}  # tid -> {"image": crop_bgr, "timestamp": seconds}
    dwell = {}        # tid -> frame_count_inside

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video.", None, None, "Metrics not available."

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Write to a fixed local file
    out_path = "results.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # dwell threshold (seconds) for loitering:
    LOITER_T = 30.0

    for i in range(N if N > 0 else 10**9):  # handle streams with unknown frame_count
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # RF-DETR expects RGB; COCO "person" == class 0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detr_model.predict(rgb, threshold=0.5)
        people = dets[dets.class_id == 1]
        tracks = tracker.update_with_detections(people)

        # Draw ROI
        if roi_poly is not None:
            cv2.polylines(frame, [roi_poly], True, (0, 255, 255), 2)

        # Track + dwell
        for bbox, tid in zip(tracks.xyxy, tracks.tracker_id):
            x1, y1, x2, y2 = bbox.astype(int)
            bottom_center = (float(x1 + x2) / 2, float(y2))

            inside = False
            if roi_poly is not None:
                inside = cv2.pointPolygonTest(roi_poly.astype(np.float32),
                                             bottom_center, False) >= 0

            if inside:
                dwell[tid] = dwell.get(tid, 0) + 1
                secs = dwell[tid] / fps

                color = (0, 255, 0)
                label = f"ID {tid} | {secs:.1f}s"
                if secs >= LOITER_T:
                    color = (0, 0, 255)
                    label += "  Loitering!"
                    if tid not in loiter_data:
                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
                        loiter_data[tid] = {"image": crop, "timestamp": i / fps}

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    elapsed = time.time() - start
    eff_fps = ((N if N > 0 else 0) / elapsed) if elapsed > 0 else 0.0

    # Prepare outputs
    gallery = []
    ts_lines = []
    for tid, data in loiter_data.items():
        if data["image"].size > 0:
            gallery.append(cv2.cvtColor(data["image"], cv2.COLOR_BGR2RGB))
        ts_lines.append(f"ID {tid}: ~{data['timestamp']:.2f}s")

    metrics = (
        f"**Processing Time:** {elapsed:.2f}s  \n"
        f"**Total FPS:** {eff_fps:.2f}  \n"
        f"**Frames:** {N}  \n"
        f"**Device:** {device}"
    )

    # Return a gr.File object to force the browser to reload the file
    return gr.File(out_path), gallery, "\n".join(ts_lines), metrics

# ------------------ UI ------------------
with gr.Blocks(title="Loitering Detection (Click-to-Polygon ROI)") as demo:
    # Persistent states
    state_video_path = gr.State(DEFAULT_VIDEO_PATH)
    state_base_frame = gr.State(None)    # original first frame (RGB, numpy)
    state_roi_points = gr.State([])      # [(x,y), ...] in ORIGINAL coords

    gr.Markdown("# Loitering Detection with RF-DETR (Pick ROI by clicking points)")

    # add text
    gr.Markdown("### Instructions: Select video, then click 'Load First Frame'. Click on the image to add ROI points (min 3). Use 'Undo' or 'Clear' as needed. Click 'Finish ROI' to close the polygon. Finally, click 'Load Detection Model' and 'Analyze Video' to start processing.")  

    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Input Video", value=DEFAULT_VIDEO_PATH, height=360)
            load_frame_btn = gr.Button("Load First Frame")

            # IMPORTANT: type="numpy", image_mode="RGB"
            roi_image = gr.Image(
                label="Click to add ROI points",
                interactive=True,
                type="numpy",
                image_mode="RGB",
                height=420,
            )

            with gr.Row():
                undo_btn = gr.Button("Undo Point")
                clear_btn = gr.Button("Clear")
                finish_btn = gr.Button("Finish ROI")

        with gr.Column(scale=1):
            processed_video = gr.Video(label="Processed Video", height=360)
            gallery = gr.Gallery(label="Loitering People", height=220, show_share_button=False)
            timestamps = gr.Textbox(label="Timestamps / IDs")
            metrics_md = gr.Markdown()
            run_btn = gr.Button("Analyze Video", variant="primary")

    status_box = gr.Textbox(label="Status", value="Click 'Load Detection Model' first.", interactive=False)
    load_btn = gr.Button("Load Detection Model")

    # Wire model load
    load_btn.click(load_detr_model, inputs=None, outputs=status_box)

    # Load first frame → set states
    load_frame_btn.click(
        get_first_frame,
        inputs=video_input,
        outputs=[roi_image, state_video_path, state_base_frame, status_box, state_roi_points]
    )

    # Corrected 'select' event handler
    roi_image.select(
        handle_click,
        inputs=[state_base_frame, state_roi_points],
        outputs=[roi_image, state_roi_points, status_box]
    )

    # Undo / Clear / Finish
    undo_btn.click(
        undo_point,
        inputs=[state_base_frame, state_roi_points],
        outputs=[roi_image, state_roi_points, status_box]
    )
    clear_btn.click(
        clear_points,
        inputs=[state_base_frame],
        outputs=[roi_image, state_roi_points, status_box]
    )
    finish_btn.click(
        finish_roi,
        inputs=[state_base_frame, state_roi_points],
        outputs=[roi_image, state_roi_points, status_box]
    )

    # Run processing
    run_btn.click(
        process_and_display_video,
        inputs=[state_video_path, state_roi_points],
        outputs=[processed_video, gallery, timestamps, metrics_md]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()