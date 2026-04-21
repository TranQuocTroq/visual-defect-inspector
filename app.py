import sys
import threading
import uvicorn
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.api import app as fastapi_app
from src.ui import demo


def run_api():
    """Run FastAPI in background thread."""
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    # Start API in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Start Gradio UI (blocking)
    demo.launch(server_name="0.0.0.0", server_port=7860)