import cv2
import numpy as np
import tqdm
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import time
import psutil
import json
import pynvml # This library comes from the `nvidia-ml-py` package

# Define the paths for the input and output videos
SOURCE_VIDEO_PATH = "../faster.mp4"
TARGET_VIDEO_PATH = "humans_detected.mp4"
PERFORMANCE_LOG_PATH = "performance_log.json"

# Get the class ID for 'person' from the COCO dataset.
PERSON_CLASS_ID = 1

# A global list to store the user-selected points for the ROI
roi_points = []
drawing = False

# Mouse callback function to get points for the polygon
def draw_polygon(event, x, y, flags, param):
    global roi_points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Function to get GPU information using pynvml
def get_gpu_info():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used_mb = gpu_memory_info.used / (1024 * 1024)
        gpu_memory_total_mb = gpu_memory_info.total / (1024 * 1024)
        pynvml.nvmlShutdown()
        return {
            "name": gpu_name,
            "utilization_percent": gpu_utilization,
            "memory_used_mb": gpu_memory_used_mb,
            "memory_total_mb": gpu_memory_total_mb,
            "supported": True
        }
    except pynvml.NVMLError:
        return {"supported": False, "message": "GPU monitoring not supported or drivers not found."}

# --- PERFORMANCE METRICS COLLECTION ---

# Record initial system state
initial_ram = psutil.virtual_memory().percent
initial_cpu = psutil.cpu_percent()
initial_gpu = get_gpu_info()
start_time = time.time()
processed_frames = 0

# Load the RF-DETR model and optimize for inference
model = RFDETRBase()
model.optimize_for_inference()

# Create a VideoCapture object to read the input video
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

# Read the first frame to get video info and let the user define the ROI
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame of the video.")
    cap.release()
    exit()

# Create a window to display the first frame and get user input
cv2.namedWindow("Select ROI - Press 'c' to confirm")
cv2.setMouseCallback("Select ROI - Press 'c' to confirm", draw_polygon, first_frame)

# Display the first frame and wait for the user to select points
print("Click points on the image to define your polygon. Press 'c' to continue.")

while True:
    temp_frame = first_frame.copy()
    if len(roi_points) > 1:
        cv2.polylines(temp_frame, [np.array(roi_points, dtype=np.int32)], False, (0, 255, 0), 2)
    cv2.imshow("Select ROI - Press 'c' to confirm", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        break
    elif key == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        exit()

cv2.destroyAllWindows()

# Convert the list of points to a NumPy array for processing
if len(roi_points) < 3:
    print("Warning: ROI requires at least 3 points. Processing all frames without ROI filtering.")
    roi_polygon = None
else:
    roi_polygon = np.array(roi_points, dtype=np.int32)
    print(f"ROI polygon confirmed with {len(roi_points)} points.")

# Reset video capture to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Get video properties for the output video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, fps, (width, height))

# Initialize the ByteTrack tracker
tracker = sv.ByteTrack()

# A dictionary to store the dwell time for each tracked ID
dwell_times = {}
loiters = {}
# Process each frame in the video
for _ in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing video frames"):
    ret, frame = cap.read()
    if not ret:
        break

    if frame is None:
        print("Warning: Skipping a frame because it is empty.")
        continue
    
    # Track the number of frames processed
    processed_frames += 1

    # Convert the frame to PIL Image format for the model
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run model prediction on the frame
    detections = model.predict(image, threshold=0.5)

    # Filter detections to only include humans (persons)
    human_detections = detections[detections.class_id == PERSON_CLASS_ID]
    
    # Update the tracker with the new detections
    tracks = tracker.update_with_detections(human_detections)

    # Annotate the frame using only OpenCV functions
    # FIX: Use zip to iterate over the tracked bounding boxes and their corresponding IDs.
    for bbox, tracker_id in zip(tracks.xyxy, tracks.tracker_id):
        # Get bounding box coordinates and cast to integers
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Calculate the bottom-center point of the bounding box
        bottom_center_x = (x1 + x2) // 2
        bottom_center_y = y2

        # Check if the bottom-center of the detection is inside the ROI polygon
        if roi_polygon is not None and cv2.pointPolygonTest(roi_polygon, (float(bottom_center_x), float(bottom_center_y)), False) >= 0:
            # If the track is new to the ROI, initialize its dwell time
            if tracker_id not in dwell_times:
                dwell_times[tracker_id] = 0
            
            # Increment the dwell time for the current track ID
            dwell_times[tracker_id] += 1
            
            color = (0, 255, 0) # Green color for bounding box
            # Create the label text with ID and dwell time
            # The dwell time is in frames, so we can convert it to seconds if we know the FPS
            seconds = dwell_times[tracker_id] / fps
            label = f"ID: {tracker_id} | Dwell: {seconds:.1f}s"
            if seconds >= 30.0:
                color = (0, 0, 255)
                label += " Loitering!"
                # save an image of the loitering person into dictionary to prevent duplicate id
                if tracker_id not in loiters:
                    loiters[tracker_id] = frame[y1:y2, x1:x2]
                    # Save the loitering person's image
                    cv2.imwrite(f"loitering_id_{tracker_id}.jpg", loiters[tracker_id])

            # Draw the bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Put the label text on the frame
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Also draw the ROI polygon on every frame for context
    if roi_polygon is not None:
        cv2.polylines(frame, [roi_polygon], True, (0, 255, 255), 2)

    # Write the annotated frame to the output video file
    out.write(frame)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
print(f"Video processing complete. Annotated video saved to {TARGET_VIDEO_PATH}")

# --- FINAL PERFORMANCE LOGGING ---

end_time = time.time()
total_time = end_time - start_time
final_ram = psutil.virtual_memory().percent
final_cpu = psutil.cpu_percent()
final_gpu = get_gpu_info()

performance_data = {
    "device_hardware": {
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "gpu_info": final_gpu
    },
    "processing_metrics": {
        "frames_processed": processed_frames,
        "total_processing_time_seconds": total_time,
        "frames_per_second": processed_frames / total_time if total_time > 0 else 0,
        "latency_per_frame_ms": (total_time / processed_frames) * 1000 if processed_frames > 0 else 0
    },
    "resource_usage": {
        "start_of_process": {
            "ram_percent": initial_ram,
            "cpu_percent": initial_cpu,
        },
        "end_of_process": {
            "ram_percent": final_ram,
            "cpu_percent": final_cpu,
        }
    }
}

# Save the performance data to a JSON file
with open(PERFORMANCE_LOG_PATH, 'w') as f:
    json.dump(performance_data, f, indent=4)

print(f"Performance metrics saved to {PERFORMANCE_LOG_PATH}")
