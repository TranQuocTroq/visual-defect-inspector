import gradio as gr
import requests
import base64
import numpy as np
import cv2


API_URL = "http://127.0.0.1:8000"


def inspect(image: np.ndarray):
    """Send image to API, return annotated image and detection summary."""
    if image is None:
        return None, "No image uploaded."

    # Convert numpy array to bytes for API request
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    img_bytes = buffer.tobytes()

    # Send to API
    response = requests.post(
        f"{API_URL}/inspect",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
    )

    if response.status_code != 200:
        return None, "API error. Make sure the server is running."

    data = response.json()

    # Decode annotated image from base64
    img_data  = base64.b64decode(data["image_base64"])
    img_array = np.frombuffer(img_data, np.uint8)
    annotated = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Build summary text
    count   = data["defect_count"]
    summary = f"Defects found: {count}\n\n"

    if count == 0:
        summary += "No defects detected — product passed QC."
    else:
        summary += "Defects detected:\n"
        for i, det in enumerate(data["detections"]):
            box   = [round(v) for v in det["box"]]
            score = det["score"]
            summary += f"  Defect {i+1}: confidence {score:.0%}, location {box}\n"

    return annotated, summary


with gr.Blocks(title="Visual Defect Inspector") as demo:
    gr.Markdown("# Visual Defect Inspector")
    gr.Markdown("Upload a product image to detect surface defects using AI.")

    with gr.Row():
        input_image  = gr.Image(label="Input image", type="numpy", height=400)
        output_image = gr.Image(label="Inspection result", height=400)

    output_text = gr.Textbox(label="Detection summary", lines=4)
    btn         = gr.Button("Inspect", variant="primary")

    btn.click(
        fn=inspect,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

    gr.Examples(
        examples=[
            ["data/samples/sample_bottle_broken_large.png"],
            ["data/samples/sample_bottle_broken_small.png"],
            ["data/samples/sample_bottle_contamination.png"],
            ["data/samples/sample_metal_nut_bent.png"],
            ["data/samples/sample_metal_nut_scratch.png"],
            ["data/samples/sample_bottle_good.png"],
        ],
        inputs=input_image,
        label="Sample images — click to test"
    )

if __name__ == "__main__":
    demo.launch()