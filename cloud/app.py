import gradio as gr
import torch
import time
import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText

# Global variables for the model and processor
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

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
            return f"Model loaded successfully on {device_name}. Ready for inference."
        except Exception as e:
            return f"Error loading model: {e}"
    return "Model is already loaded."

def run_inference(video_path, query):
    """
    Runs the VLM inference on the video and returns the generated text and performance metrics.
    Gradio automatically handles the threading for this function.
    """
    global model, processor
    if model is None or processor is None:
        return "Model not loaded. Please wait or check the logs.", ""

    try:
        # Get video properties for FPS calculation
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Start timing
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
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        fps = frame_count / processing_time if processing_time > 0 else 0

        metrics = f"**Processing Time:** {processing_time:.2f} seconds\n**Est. FPS:** {fps:.2f}\n**Device:** {device}"
        
        return response, metrics

    except Exception as e:
        return f"An error occurred during inference: {e}", "Metrics not available due to error."

# --- Gradio UI setup ---
with gr.Blocks(title="VLM Video Analysis") as demo:
    gr.Markdown("# VLM Question Answering on Video for Panoptic")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Video Player")
            # Gradio's Video component handles playback from a file path
            video_input = gr.Video(label="Input Video", value="../faster.mp4")
        
        with gr.Column(scale=1):
            gr.Markdown("### Model Interaction")
            query_input = gr.Textbox(label="Enter your query:", placeholder="e.g., How many people stood in the center?")
            
            # The Textbox will display the model's output
            response_output = gr.Textbox(label="Model Response:", interactive=False)
            
            # The Textbox for performance metrics
            performance_metrics_output = gr.Textbox(label="Performance Metrics:", interactive=False)
            
            # The button to trigger the inference
            submit_button = gr.Button("Analyze Video")
    
    # Add gr.Examples to provide sample inputs
    gr.Examples(
        examples=[
            ["faster.mp4", "How many people are loitering for more then 30 seconds?"],
            ["fence.mp4", "Any fence jumping or climbing? and what does the person look like? "]
        ],
        inputs=[video_input, query_input]
    )

    # Load the model and show a status message
    gr.Markdown("---")
    status_box = gr.Textbox(label="Status", value="Model is being loaded...", interactive=False)
    load_model_btn = gr.Button("Check Model Status")
    load_model_btn.click(fn=load_model, inputs=None, outputs=status_box)

    # When the submit button is clicked, call the inference function
    submit_button.click(
        fn=run_inference,
        inputs=[video_input, query_input],
        outputs=[response_output, performance_metrics_output],
    )

# Launch the Gradio application
demo.launch()