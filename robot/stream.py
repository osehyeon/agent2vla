import cv2, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
idx = sys.argv[1]; idx = int(idx) if idx.isdigit() else idx; port = int(sys.argv[2])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK, open_camera
D = str(WORK / "webcam"); os.makedirs(D, exist_ok=True); FLAG = D + "/REC"
cap = open_camera(idx, 1920, 1080)
frame = [None]; state = {"w": None, "path": None, "n": 0}
def loop():
    t0, cnt, fps = time.time(), 0, 30.0
    while True:
        ok, f = cap.read()
        if not ok: continue
        cnt += 1
        if cnt == 60: fps = cnt / (time.time() - t0)
        frame[0] = cv2.imencode(".jpg", cv2.resize(f, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        rec = os.path.exists(FLAG) and time.time() < float(open(FLAG).read().strip() or 0)
        if rec and state["w"] is None:
            state["path"] = time.strftime(f"{D}/webcam_%Y%m%d_%H%M%S.mp4")
            state["w"] = cv2.VideoWriter(state["path"], cv2.VideoWriter_fourcc(*"mp4v"), round(fps), (f.shape[1], f.shape[0])); state["n"] = 0
            print("REC start", state["path"], f.shape, round(fps), flush=True)
        if rec: state["w"].write(f); state["n"] += 1
        elif state["w"] is not None:
            state["w"].release(); print("REC stop", state["path"], state["n"], "frames", flush=True); state["w"] = None
            if os.path.exists(FLAG): os.remove(FLAG)
threading.Thread(target=loop, daemon=True).start()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path != "/":
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
        try:
            while True:
                if frame[0] is not None:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + frame[0] + b"\r\n")
                time.sleep(1 / 20)
        except (BrokenPipeError, ConnectionResetError): pass
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
