import torch
import time
import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
import psutil
import pynvml
import json

# Global variables for the model and processor
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def load_model():
    """Loads the VLM model and processor."""
    global model, processor, device
    if model is None:
        try:
            model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            print(f"Model loaded successfully on {device_name}. Ready for inference.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

def run_inference(video_path, query):
    """
    Runs the VLM inference on the video and returns the generated text and frame count.
    """
    global model, processor
    if model is None or processor is None:
        print("Model not loaded. Please load the model first.")
        return None, None

    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start_time = time.time()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": query}
                ]
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        response = generated_texts[0]
        
        end_time = time.time()
        
        return response, frame_count, start_time, end_time

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None, None, None, None

if __name__ == "__main__":
    SOURCE_VIDEO_PATH = "../faster.mp4"
    QUERY = "How many people are loitering for more than 30 seconds?"

    # Capture initial resource usage
    initial_ram = psutil.virtual_memory().percent
    initial_cpu = psutil.cpu_percent()
    initial_gpu = get_gpu_info()
    
    if load_model():
        print("Starting video analysis...")
        response, frame_count, start_time, end_time = run_inference(SOURCE_VIDEO_PATH, QUERY)
        
        if response:
            # Capture final resource usage
            final_ram = psutil.virtual_memory().percent
            final_cpu = psutil.cpu_percent()
            final_gpu = get_gpu_info()

            # Calculate metrics
            total_processing_time = end_time - start_time
            fps = frame_count / total_processing_time if total_processing_time > 0 else 0
            latency_per_frame_ms = (total_processing_time / frame_count) * 1000 if frame_count > 0 else 0

            # Construct the final data dictionary
            performance_data = {
                "device_hardware": {
                    "cpu_count": psutil.cpu_count(logical=False),
                    "logical_cpu_count": psutil.cpu_count(logical=True),
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "gpu_info": initial_gpu
                },
                "processing_metrics": {
                    "frames_processed": frame_count,
                    "total_processing_time_seconds": total_processing_time,
                    "frames_per_second": fps,
                    "latency_per_frame_ms": latency_per_frame_ms
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

            print("\n--- Analysis Result ---")
            print(f"Query: {QUERY}")
            print(f"Response: {response}")
            
            print("\n--- Performance Metrics ---")
            print(json.dumps(performance_data, indent=4))
