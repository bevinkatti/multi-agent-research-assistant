import threading
import time
import os

def run_api():
    os.system("uvicorn api.main:app --host 0.0.0.0 --port 8000")

def run_frontend():
    time.sleep(15)
    os.system(
        "streamlit run frontend/app.py "
        "--server.port 7860 "
        "--server.address 0.0.0.0 "
        "--server.headless true"
    )

if __name__ == '__main__':
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    run_frontend()