"""Run a trained SmolVLA policy on the SO-101 follower.

usage: python rollout.py --policy CKPT --task "Move the cube to region 2." [--duration 25] [--view HOST:PORT]
Starts from the same HOME pose as the data collection, runs the policy for up to --duration seconds,
then returns HOME and folds. Ctrl+C stops the policy early; the arm still returns HOME and folds.
--view streams both cameras to viewer.py on HOST:PORT; Esc or q there stops the policy. Opens only the two robot cameras (robot_cams.json).
"""
import argparse, json, os, signal, socket, struct, sys, threading, time
from pathlib import Path
import cv2, numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK, FOLLOWER_PORT, FOLLOWER_ID, robot_cams, open_camera, robot_lock
from kin import Arm

ap = argparse.ArgumentParser(description="Run a trained SmolVLA policy on the SO-101 follower.")
ap.add_argument("--policy", default=None, help="checkpoint folder or Hugging Face repo id")
ap.add_argument("--task", default="", help="instruction text")
ap.add_argument("--duration", type=float, default=25, help="seconds to run the policy")
ap.add_argument("--tag", default=time.strftime("%H%M%S"), help="output folder name under rollouts/")
ap.add_argument("--look", default=None, help="only go HOME, save a top image to this path, and fold (no policy)")
ap.add_argument("--hold", type=float, default=0, help="with --look, seconds to stay at HOME before folding (Ctrl+C folds early)")
ap.add_argument("--view", default=None, help="HOST:PORT of a running viewer.py")
A = ap.parse_args(); CK, TASK, SECS, TAG = A.policy, A.task, A.duration, A.tag
if not A.look and not (TASK and CK): ap.error("--policy and --task are required unless --look is given")
stop_req = []


def _sigint(*_):
    if not stop_req: print("\nCtrl+C: stopping the policy, the arm returns HOME and folds", flush=True)
    else: print("\nalready stopping; unplug the 12 V supply for a hard stop", flush=True)
    stop_req.append(1)
WD = WORK; OUT = WD / "rollouts" / TAG; OUT.mkdir(parents=True, exist_ok=True)
FPS, MAX_STEP_DEG, CONTACT_DEG = 30, 5.0, 15.0
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


class Cam:
    def __init__(self, idx):
        self.c = open_camera(idx)
        self.f, self.run = None, True
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while self.f is None:
            if time.time() - t0 > 5: raise RuntimeError(f"camera {idx} gives no frames")
            time.sleep(0.05)
        if self.f.shape[1] != 1280: raise RuntimeError(f"camera {idx} is not a 720p robot camera")
    def _loop(self):
        while self.run:
            ok, f = self.c.read()
            if ok: self.f = f
    def get(self): return self.f.copy()
    def close(self): self.run = False; time.sleep(0.1); self.c.release()


fixed = robot_cams()
robot_lock()
fol = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)); fol.connect(calibrate=False)
M = list(fol.bus.motors); cal = fol.bus.calibration
for m in M:
    if m != "gripper": fol.bus.write("Torque_Limit", m, 1000)
lo = {m: -(c.range_max - c.range_min) * 360 / 4095 / 2 * 0.97 for m, c in cal.items()}
hi = {m: -v for m, v in lo.items()}; lo["gripper"], hi["gripper"] = 0.0, 100.0
mid = {m: (2047 - (c.range_min + c.range_max) / 2) * 360 / 4095 for m, c in cal.items() if m != "gripper"}
HOME = {**mid, "shoulder_pan": -85, "wrist_flex": mid["wrist_flex"] + 60, "wrist_roll": 0, "gripper": 3}
REST = {**mid, "shoulder_lift": -64, "elbow_flex": 65, "wrist_flex": 0, "wrist_roll": 0, "gripper": 3}
o = fol.get_observation(); last = np.array([o[f"{m}.pos"] for m in M])
arm = Arm(str(HERE / "so101.urdf"))


def workspace_violation(q):
    """None if the commanded pose is inside the safe workspace, else a reason string."""
    tip, _, G = arm.fk(dict(zip(M, q)))
    x, y, z = tip * 1000; r = np.hypot(x, y); pan = q[0]
    wrist = G[:3, 3] * 1000
    if z < -25: return f"tip below table (z {z:.0f} mm)"
    if r < 150 and z < 120: return f"tip too close to body (r {r:.0f} mm)"
    if pan > 55 and r < 220: return f"stand-side keep-out (pan {pan:.0f}, r {r:.0f})"
    if np.hypot(*wrist[:2]) > 330: return f"beyond reach (wrist r {np.hypot(*wrist[:2]):.0f})"
    return None


def present():
    o = fol.get_observation(); return np.array([o[f"{m}.pos"] for m in M])


def go(t, secs):
    global last
    b = np.array([t.get(m, v) for m, v in zip(M, last)]); a = last.copy()
    secs = max(secs, np.abs(b - a)[:5].max() / 40.0)
    for s in np.linspace(0, 1, max(2, int(secs * FPS))):
        s = 0.5 - 0.5 * np.cos(np.pi * s); fol.send_action({f"{m}.pos": v for m, v in zip(M, a + (b - a) * s)}); time.sleep(1 / FPS)
    last = b


def cur(): return dict(zip(M, last))


def to_home():
    go({**cur(), "wrist_roll": 0, "gripper": 3}, 0.8)
    q = cur()
    if q["elbow_flex"] > mid["elbow_flex"] + 15 or q["shoulder_lift"] > mid["shoulder_lift"] + 15:
        go({**cur(), "shoulder_lift": mid["shoulder_lift"], "elbow_flex": mid["elbow_flex"], "wrist_flex": mid["wrist_flex"] + 60}, 1.5)
    go(HOME, 2.0)


def fold():
    go({**cur(), "gripper": 50}, 0.5); go({**cur(), "wrist_roll": 0}, 0.8)
    go({**cur(), **{k: mid[k] for k in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]}}, 2.0)
    for m in ["shoulder_lift", "wrist_flex", "elbow_flex"]: go({**cur(), m: REST[m]}, 1.6)
    go({**cur(), "gripper": 3}, 0.4)


class Timeout(Exception): pass
def _alarm(*_): raise Timeout("timeout")
signal.signal(signal.SIGALRM, _alarm); signal.signal(signal.SIGINT, _sigint)

class View:
    """Streams the two cameras to viewer.py; a b'S' back from it requests a stop."""
    def __init__(self, addr):
        h, p = addr.rsplit(":", 1); self.s = socket.create_connection((h, int(p)), timeout=3); self.s.settimeout(None)
        self.status, self.run = "starting", True
        threading.Thread(target=self._send, daemon=True).start(); threading.Thread(target=self._recv, daemon=True).start()
    def _send(self):
        while self.run:
            try:
                if "top" in cams and "wrist" in cams:
                    im = np.hstack([cv2.resize(cams[n].get(), (640, 360)) for n in ("top", "wrist")])
                    cv2.putText(im, self.status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    b = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                    self.s.sendall(struct.pack(">I", len(b)) + b)
            except OSError: return
            time.sleep(0.1)
    def _recv(self):
        try:
            while self.run and self.s.recv(1) == b"S":
                if not stop_req: log("stop requested from the viewer")
                stop_req.append(1)
        except OSError: pass
    def close(self):
        self.run = False
        try: self.s.close()
        except OSError: pass


cams = {}; rows = []; stop = "done"; view = None
try:
    if not A.look:
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pol = SmolVLAPolicy.from_pretrained(CK).to(dev).eval()
        pre, post = make_pre_post_processors(pol.config, CK, preprocessor_overrides={"device_processor": {"device": dev}})
        log("policy loaded on", dev)
    cams = {"top": Cam(fixed["top"]), "wrist": Cam(fixed["wrist"])}
    if A.view:
        try: view = View(A.view); view.status = "going home"
        except OSError as ex: log("viewer not reachable:", ex)
    for m in ["elbow_flex", "wrist_flex", "shoulder_lift"]:
        if abs(cur()[m] - mid[m]) > 3: go({**cur(), m: mid[m]}, 1.5)
    to_home(); time.sleep(0.5)

    if A.look:
        time.sleep(0.5); cv2.imwrite(A.look, cams["top"].get()); stop = "look"; log("saved", A.look)
        t_hold = time.time()
        if A.hold: log(f"holding HOME for {A.hold:.0f}s")
        while time.time() - t_hold < A.hold and not stop_req: time.sleep(0.1)
    else:
        def obs_now(state):
            d = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": TASK}
            for n, key in [("top", "observation.images.camera1"), ("wrist", "observation.images.camera2")]:
                im = cv2.cvtColor(cv2.resize(cams[n].get(), (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                d[key] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            return d

        if (WD / "STOP").exists(): (WD / "STOP").unlink()
        pol.reset(); signal.alarm(int(SECS + 60)); t0 = time.time(); streak = 0; cmd = last.copy(); k = 0
        log(f"run '{TASK}' for {SECS:.0f}s")
        while time.time() - t0 < SECS:
            if stop_req:
                stop = "stop request"; break
            if (WD / "STOP").exists():
                stop = "STOP file"; (WD / "STOP").unlink(); break
            ts = time.time(); st = present()
            err = np.abs(cmd - st)[:5]
            streak = streak + 1 if err.max() > CONTACT_DEG else 0
            if streak >= 3:
                stop = f"contact/tracking error {err.round(1).tolist()}"; break
            with torch.no_grad():
                a = post(pol.select_action(pre(obs_now(st)))).reshape(-1).cpu().numpy()[:len(M)]
            tgt = np.array([np.clip(a[i], lo[m], hi[m]) for i, m in enumerate(M)])
            step = np.clip(tgt - cmd, -MAX_STEP_DEG, MAX_STEP_DEG); step[5] = tgt[5] - cmd[5]
            nxt = cmd + step
            why = workspace_violation(nxt)
            if why is not None:
                stop = f"workspace guard: {why}"; break
            cmd = nxt
            fol.send_action({f"{m}.pos": v for m, v in zip(M, cmd)}); last = cmd.copy()
            rows.append(np.r_[time.time() - t0, st, a, cmd])
            if view: view.status = f"{TASK}  policy {time.time() - t0:4.1f}/{SECS:.0f} s"
            if k % 30 == 0:
                cv2.imwrite(str(OUT / f"top_{k:04d}.jpg"), cams["top"].get()); cv2.imwrite(str(OUT / f"wrist_{k:04d}.jpg"), cams["wrist"].get())
            k += 1
            dt = 1 / FPS - (time.time() - ts)
            if dt > 0: time.sleep(dt)
        signal.alarm(0)
        log("policy run ended:", stop, f"{k} steps")
except BaseException as ex:
    stop = f"error: {ex!r}"; log(stop)
finally:
    signal.alarm(0)
    np.save(OUT / "log.npy", np.array(rows)) if rows else None
    json.dump({"task": TASK, "ckpt": CK, "secs": SECS, "stop": stop, "steps": len(rows)}, open(OUT / "meta.json", "w"))
    signal.alarm(40)
    try:
        if view: view.status = f"stopped: {stop} - returning home"
        to_home()
        if view: view.status = "folding"
        if cams: cv2.imwrite(str(OUT / "final_top.jpg"), cams["top"].get())
        fold()
    except BaseException as ex: log("return error", repr(ex))
    signal.alarm(0)
    for m in M:
        for _ in range(5):
            try: fol.bus.write("Torque_Enable", m, 0); break
            except Exception: time.sleep(0.3)
    try: fol.bus.port_handler.closePort()
    except Exception: pass
    if view: view.close()
    for c in cams.values(): c.close()
    log("END", OUT)
