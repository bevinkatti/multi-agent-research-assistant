import subprocess
import sys
import time
import urllib.request


API_HOST = "127.0.0.1"
API_PORT = 8000
FRONTEND_PORT = 7860


def start_api():
    print("Starting FastAPI backend...", flush=True)

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(API_PORT),
        ]
    )


def wait_for_api(api_process, timeout=120):
    print("Waiting for FastAPI backend...", flush=True)

    health_url = f"http://{API_HOST}:{API_PORT}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:

        # If Uvicorn crashed, stop waiting immediately.
        if api_process.poll() is not None:
            print(
                f"ERROR: FastAPI process exited with code "
                f"{api_process.returncode}",
                flush=True,
            )
            return False

        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    print("FastAPI backend is ready.", flush=True)
                    return True

        except Exception:
            pass

        time.sleep(2)

    print(
        f"ERROR: FastAPI did not become ready within {timeout} seconds.",
        flush=True,
    )
    return False


def start_frontend():
    print("Starting Streamlit frontend...", flush=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.port",
            str(FRONTEND_PORT),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
        ]
    )


if __name__ == "__main__":
    api_process = start_api()

    if wait_for_api(api_process):
        start_frontend()
    else:
        api_process.terminate()
        sys.exit(1)