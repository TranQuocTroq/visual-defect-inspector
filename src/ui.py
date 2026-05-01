import gradio as gr
import cv2
import os
from src.detector import DefectDetector

# Path resolution
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# CSS to force square aspect ratio and remove black empty space
custom_css = """
#live-monitor {
    aspect-ratio: 1 / 1 !important;
    width: 100% !important;
    height: auto !important;
}
#live-monitor .image-container, #live-monitor img {
    height: 100% !important;
    width: 100% !important;
    object-fit: contain !important;
}
"""

detector = DefectDetector(MODELS_DIR)

def process_video_stream(video_path, product_name):
    if not video_path:
        yield None, [], "Error: No input video. Please upload or select a sample."
        return

    cap = cv2.VideoCapture(video_path)
    gallery = []
    total, defects = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        total += 1
        status, annotated_frame, is_defect = detector.inspect(frame, product_name)
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        if is_defect:
            defects += 1
            if len(gallery) < 15:
                gallery.append((img_rgb.copy(), status))
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
            
        cv2.putText(img_rgb, status, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        
        yield_rate = ((total - defects) / total) * 100 if total > 0 else 0
        report = (
            f"FACTORY ANALYTICS REPORT\n"
            f"Product: {product_name.upper()}\n"
            f"Total Inspected: {total}\n"
            f"Defects Blocked: {defects}\n"
            f"Production Yield: {yield_rate:.2f}%"
        )
        
        yield img_rgb, gallery, report

    cap.release()

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Industrial Quality Control System (Hybrid AI)")
    
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            product_type = gr.Dropdown(
                choices=["wood", "zipper", "pill"], 
                value="wood", 
                label="Product Line Selection"
            )
            input_vid = gr.Video(label="Conveyor Input", height=400)
            btn = gr.Button("RUN ANALYSIS", variant="primary")
            results = gr.Textbox(label="Operational Statistics", lines=6)
        
        with gr.Column(scale=1):
            output_live = gr.Image(
                label="Live AI Insight Monitor", 
                interactive=False, 
                elem_id="live-monitor"
            )
            
    gr.Markdown("### Captured Defects Gallery")
    gallery = gr.Gallery(label="Detection History", columns=5, height="auto")
    
    # Disable auto-API generation for the button to prevent Gradio schema parsing crash
    btn.click(
        fn=process_video_stream, 
        inputs=[input_vid, product_type], 
        outputs=[output_live, gallery, results],
        api_name=False
    )

    # Display sample videos dynamically from the data/samples folder
    example_folder = os.path.join(ROOT_DIR, "data", "samples")
    if os.path.exists(example_folder):
        vid_files = [[os.path.join(example_folder, f)] for f in os.listdir(example_folder) if f.endswith('.mp4')]
        if vid_files:
            gr.Examples(
                examples=vid_files, 
                inputs=input_vid,
                label="Or click on an available sample video below to run the demo:",
                api_name=False
            )