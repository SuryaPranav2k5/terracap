"""GeoCaption Real-Time Python Inference Microservice.

Runs a fast, local HTTP server on port 5001 keeping the PyTorch model
resident in memory for real-time sub-second inference.
"""

import sys
import json
import time
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from infer import load_model_and_vocab, run_inference_on_pil, parse_image_arg

PORT = 5001
HOST = "127.0.0.1"


class InferenceHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "model": "NWPUCaptioner"}).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            image_input = data.get("image") or data.get("image_base64")
            if not image_input:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing 'image' parameter"}).encode("utf-8"))
                return

            beam_size = int(data.get("beam_size", 3))
            max_len = int(data.get("max_len", 30))

            pil_img = parse_image_arg(image_input)
            result = run_inference_on_pil(pil_img, beam_size=beam_size, max_len=max_len)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, format, *args):
        # Concise logging
        try:
            sys.stdout.write(f"[Backend] {format % args}\n")
            sys.stdout.flush()
        except Exception:
            pass


def run_service(port: int = PORT, host: str = HOST):
    print(f"Pre-warming GeoCaption PyTorch model on {host}:{port} ...")
    load_model_and_vocab()
    server = ThreadingHTTPServer((host, port), InferenceHandler)
    print(f"GeoCaption Inference Microservice ready on http://{host}:{port}/predict")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Inference Microservice...")
        server.server_close()


if __name__ == "__main__":
    run_service()

