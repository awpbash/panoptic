import argparse
import os
import time
import threading
from collections import deque
import base64

import cv2
import numpy as np
import requests


def post_frame(cloud_url: str, stream_id: str, frame_bgr, conf: float, timeout=0.3, jpeg_quality=80, draw_annotations=False):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    data = {
        "stream_id": stream_id,
        "conf": str(conf),
        "draw_annotations": "true" if draw_annotations else "false",
        "return_image": "true" # Always request the image back to display it
    }
    try:
        r = requests.post(f"{cloud_url.rstrip('/')}/ingest", data=data, files=files, timeout=timeout)
        if r.ok:
            return r.json()
        print(f"[ERROR] API request failed: {r.status_code} - {r.text}")
        return None
    except requests.RequestException as e:
        print(f"[ERROR] Network request exception: {e}")
        return None


def downscale_gray(frame_bgr, w=320):
    h, w0 = frame_bgr.shape[:2]
    if w0 != w:
        nh = int(h * (w / float(w0)))
        frame_bgr = cv2.resize(frame_bgr, (w, nh), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return g


def motion_score(prev_gray, cur_gray):
    d = cv2.absdiff(prev_gray, cur_gray)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    return float(d.mean()), d


def contours_near_line(diff_img, line_x, margin_px=24, min_area=60):
    _, th = cv2.threshold(diff_img, 20, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = th.shape[:2]
    x0 = max(0, int(line_x - margin_px))
    x1 = min(w - 1, int(line_x + margin_px))
    if x1 <= x0:
        x1 = min(w - 1, x0 + 1)
    band = (x0, x1)

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if (x <= band[1]) and (x + cw >= band[0]):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Webcam/RTSP/File → Cloud People Counter (smart sampling)")
    ap.add_argument("--source", default=os.getenv("SOURCE", "0"),
                    help="0/1... for webcam, path to file, or RTSP URL")
    ap.add_argument("--cloud", default=os.getenv("CLOUD_URL", "http://localhost:8000"),
                    help="Cloud API base URL (expects POST /ingest)")
    ap.add_argument("--stream-id", default=os.getenv("STREAM_ID", "cam1"),
                    help="Stream identifier")
    ap.add_argument("--conf", type=float, default=float(os.getenv("CONF_THRES", "0.5")),
                    help="Detection confidence threshold sent to server")
    ap.add_argument("--resize", type=int, default=int(os.getenv("RESIZE_WIDTH", "640")),
                    help="Resize width before sending (keep aspect). 0 = no resize")
    ap.add_argument("--jpegq", type=int, default=int(os.getenv("JPEG_QUALITY", "80")),
                    help="JPEG quality for upload (1-100)")
    ap.add_argument("--ds-proc-w", type=int, default=int(os.getenv("PROC_WIDTH", "320")),
                    help="Downscaled width for motion analysis (CPU-light)")
    ap.add_argument("--motion-thresh", type=float, default=float(os.getenv("MOTION_THRESH", "3.5")),
                    help="Mean diff threshold (0-255) to consider there is motion")
    ap.add_argument("--line-frac", type=float, default=float(os.getenv("LINE_X_FRAC", "0.5")),
                    help="Vertical counting line as fraction of width [0..1] for motion band check")
    ap.add_argument("--band-margin", type=int, default=int(os.getenv("BAND_MARGIN_PX", "24")),
                    help="Half-width (in px on downscaled image) for motion band around the line")
    ap.add_argument("--min-area", type=int, default=int(os.getenv("MIN_AREA", "60")),
                    help="Minimum contour area (downscaled) to consider as motion")
    ap.add_argument("--min-interval", type=float, default=float(os.getenv("MIN_INTERVAL", "0.15")),
                    help="Minimum seconds between two uploads (cool-down)")
    ap.add_argument("--max-interval", type=float, default=float(os.getenv("MAX_INTERVAL", "2.0")),
                    help="Maximum seconds allowed without sending (heartbeat)")
    args = ap.parse_args()

    src = args.source
    try:
        idx = int(src)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src}")

    print(f"[INFO] Source: {src}")
    print(f"[INFO] Cloud:  {args.cloud}  stream_id={args.stream_id}")
    print("[INFO] Press 'q' to quit.")

    last_counts = {"left_to_right": 0, "right_to_left": 0}
    prev_gray = None
    last_send_t = 0.0
    recent_motion = deque(maxlen=5)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        h_full, w_full = frame.shape[:2]
        
        cur_gray = downscale_gray(frame, w=args.ds_proc_w)
        h_ds, w_ds = cur_gray.shape[:2]
        line_x_ds = int(w_ds * args.line_frac)

        should_send = False
        now = time.time()

        if prev_gray is not None:
            mscore, diff_img = motion_score(prev_gray, cur_gray)
            near_line = False
            if mscore >= args.motion_thresh:
                near_line = contours_near_line(diff_img, line_x_ds,
                                               margin_px=args.band_margin,
                                               min_area=args.min_area)
            recent_motion.append(1 if (mscore >= args.motion_thresh and near_line) else 0)

            motion_recent = any(recent_motion)
            if motion_recent and (now - last_send_t) >= args.min_interval:
                should_send = True
            elif (now - last_send_t) >= args.max_interval:
                should_send = True
        else:
            pass

        if args.resize > 0:
            if w_full != args.resize:
                new_h = int(h_full * (args.resize / float(w_full)))
                frame_to_send = cv2.resize(frame, (args.resize, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_to_send = frame
        else:
            frame_to_send = frame

        display_frame = frame.copy()
        if should_send:
            resp = post_frame(args.cloud, args.stream_id, frame_to_send,
                              conf=args.conf, jpeg_quality=args.jpegq, draw_annotations=True)
            if resp and resp.get("ok"):
                counts = resp.get("counts", last_counts)
                last_counts = counts
                last_send_t = now
                if "annotated_image_b64" in resp and resp["annotated_image_b64"] is not None:
                    try:
                        img_bytes = base64.b64decode(resp["annotated_image_b64"])
                        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        display_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    except Exception as e:
                        print(f"[ERROR] Failed to decode image from server: {e}")
            
        # cv2.putText(display_frame, f"L->R: {last_counts.get('left_to_right',0)}",
        #             (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # cv2.putText(display_frame, f"R->L: {last_counts.get('right_to_left',0)}",
        #             (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Smart Sampling Client (press 'q' to quit)", display_frame)
        prev_gray = cur_gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()