"""OpenCV window for a run on another machine. usage: python viewer.py [PORT]
rollout.py --view HOST:PORT connects here and streams JPEG frames. Esc or q in the window stops the run."""
import socket, struct, sys, threading
import cv2, numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9877
frame = [None]; conn = [None]


def serve():
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT)); srv.listen(1)
    while True:
        c, _ = srv.accept(); conn[0] = c
        try:
            while True:
                head = c.recv(4, socket.MSG_WAITALL)
                if len(head) < 4: break
                n = struct.unpack(">I", head)[0]; data = c.recv(n, socket.MSG_WAITALL)
                frame[0] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            pass
        conn[0] = None; c.close()


threading.Thread(target=serve, daemon=True).start()
idle = np.full((360, 1280, 3), 40, np.uint8)
cv2.putText(idle, "waiting for the robot", (440, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
while True:
    cv2.imshow("agent2vla", frame[0] if conn[0] is not None and frame[0] is not None else idle)
    k = cv2.waitKey(30) & 0xFF
    if k in (27, ord("q")) and conn[0] is not None:
        try: conn[0].sendall(b"S")
        except OSError: pass
