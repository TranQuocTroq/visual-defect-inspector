import sys
import uvicorn
import gradio as gr
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Resolve internal imports
sys.path.append(str(Path(__file__).parent))

from src.api import app as fastapi_app
from src.ui import demo

# 1. FIX LỖI CORS: Cấp quyền để giao diện Web được phép gọi API ngầm
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Bật hàng đợi cho Gradio (Bắt buộc cho luồng Video)
demo.queue()

# 3. Gộp Gradio UI vào FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    # 4. FIX LỖI HUGGING FACE PROXY: Thêm proxy_headers và forwarded_allow_ips
    # Điều này giúp Websocket của Gradio xuyên qua được tường lửa của Hugging Face
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860, 
        proxy_headers=True, 
        forwarded_allow_ips="*"
    )