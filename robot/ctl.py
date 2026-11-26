"""Command server for direct control: Claude writes ctl/cmd.json (pose, joint, snap, fold), this moves the arm or saves camera images and reports in ctl/out.json."""
import json, os, subprocess, sys, time, traceback
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK, FOLLOWER_PORT, FOLLOWER_ID, robot_cams, open_camera, robot_lock
from kin import Arm
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

D = str(WORK / "ctl"); os.makedirs(D, exist_ok=True)
CMD, OUT = D + "/cmd.json", D + "/out.json"
for f in (CMD, OUT):
    if os.path.exists(f): os.remove(f)
robot_lock()
fol = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)); fol.connect(calibrate=False)
cal = fol.bus.calibration; M = list(fol.bus.motors)
lim = {k: (-(c.range_max - c.range_min) * 360 / 4095 / 2 * 0.97, (c.range_max - c.range_min) * 360 / 4095 / 2 * 0.97) for k, c in cal.items()}
arm = Arm(str(HERE / "so101.urdf"), lim)
o = fol.get_observation(); last = np.array([o[f"{m}.pos"] for m in M])
idx = robot_cams()
cams = {n: open_camera(idx[n]) for n in ["top", "wrist"]}
cur_xyz = None; seed = None


def go(t, secs):
    global last
    b = np.array([t.get(m, v) for m, v in zip(M, last)]); a = last.copy()
    for s in np.linspace(0, 1, max(2, int(secs * 50))):
        s = 0.5 - 0.5 * np.cos(np.pi * s); fol.send_action({f"{m}.pos": v for m, v in zip(M, a + (b - a) * s)}); time.sleep(0.02)
    last = b; time.sleep(0.3)


def obs():
    o = fol.get_observation(); return {m: round(o[f"{m}.pos"], 1) for m in M}


def solve(xyz):
    global seed
    if seed is not None:
        q, e, ang = arm.ik(xyz, seed)
        if e <= 3 and ang <= 8: seed = q; return q, e, ang
    q, e, ang = arm.ik_best(xyz); seed = q; return q, e, ang


mid = {m: (2047 - (c.range_min + c.range_max) / 2) * 360 / 4095 for m, c in cal.items() if m != "gripper"}; mid["gripper"] = 5
rest = {**mid, "shoulder_lift": -64, "elbow_flex": 78, "wrist_flex": 72, "gripper": 3}
def staged(t, order):
    for m in order: go({**dict(zip(M, last)), m: t[m]}, 1.8)

staged(mid, ["elbow_flex", "wrist_flex", "shoulder_lift", "shoulder_pan", "wrist_roll", "gripper"])
json.dump({"id": 0, "ok": True, "obs": obs()}, open(OUT, "w")); print("READY", flush=True)
try:
    while True:
        if not os.path.exists(CMD): time.sleep(0.1); continue
        try: c = json.load(open(CMD))
        except Exception: time.sleep(0.1); continue
        os.remove(CMD); r = {"id": c.get("id"), "ok": True}
        try:
            op = c["op"]
            if op == "pose":
                xyz = c["xyz"]; roll = c.get("roll", dict(zip(M, last))["wrist_roll"]); grip = c.get("grip", dict(zip(M, last))["gripper"])
                n = c.get("n", 1) if cur_xyz is not None else 1
                p0 = np.array(cur_xyz if cur_xyz is not None else xyz, float)
                for i in range(1, n + 1):
                    p = p0 + (np.array(xyz, float) - p0) * i / n
                    q, e, ang = solve(p)
                    if e > 5 or ang > 10: raise RuntimeError(f"IK fail {p.round(0).tolist()} err {e:.1f} tilt {ang:.1f}")
                    go({**q, "wrist_roll": roll, "gripper": grip}, c.get("secs", 2) / n)
                cur_xyz = list(xyz)
            elif op == "joint": go({**dict(zip(M, last)), **c["q"]}, c.get("secs", 2)); cur_xyz = None if any(k in c["q"] for k in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]) else cur_xyz
            elif op == "snap":
                for n_, cam in cams.items():
                    for _ in range(8): cam.read()
                    ok, f = cam.read()
                    if ok: cv2.imwrite(f"{D}/{c['tag']}_{n_}.jpg", f)
            elif op == "fold": break
            r["obs"] = obs(); r["xyz"] = cur_xyz
        except Exception as ex:
            r = {"id": c.get("id"), "ok": False, "err": str(ex), "obs": obs()}
        json.dump(r, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT); print("DONE", r, flush=True)
finally:
    go({**dict(zip(M, last)), "shoulder_lift": mid["shoulder_lift"], "elbow_flex": mid["elbow_flex"], "wrist_flex": mid["wrist_flex"]}, 3)
    staged(rest, ["shoulder_lift", "wrist_flex", "elbow_flex", "shoulder_pan", "wrist_roll", "gripper"]); fol.disconnect()
    for cam in cams.values(): cam.release()
    json.dump({"id": "end", "ok": True}, open(OUT, "w")); print("END", flush=True)
