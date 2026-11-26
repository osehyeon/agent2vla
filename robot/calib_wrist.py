"""Measure how the wrist camera is mounted: which way its image axes point and how many pixels a millimetre is.

usage: python calib_wrist.py
Put the cube red face up in any region first. The arm goes HOME, hovers over the cube, shifts 10 mm along
the robot x and y axes while watching the cube in the wrist camera, then folds.
Writes WORK/wrist_cam.json, which collect.py reads before its built-in values.
Not yet run on the robot; compare its output with the built-in XIMG_G, YIMG_G and pxmm in collect.py first.
"""
import json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK, FOLLOWER_PORT, FOLLOWER_ID, robot_cams, open_camera, setup_file, robot_lock
from kin import Arm
from cube import find_cube
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FPS, Z, STEP = 30, 50, 10
robot_lock()
fol = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)); fol.connect(calibrate=False)
M = list(fol.bus.motors); cal = fol.bus.calibration
lim = {k: (-(c.range_max - c.range_min) * 360 / 4095 / 2 * 0.97, (c.range_max - c.range_min) * 360 / 4095 / 2 * 0.97) for k, c in cal.items()}
mid = {m: (2047 - (c.range_min + c.range_max) / 2) * 360 / 4095 for m, c in cal.items() if m != "gripper"}
HOME = {**mid, "shoulder_pan": -85, "wrist_flex": mid["wrist_flex"] + 60, "wrist_roll": 0, "gripper": 3}
REST = {**mid, "shoulder_lift": -64, "elbow_flex": 65, "wrist_flex": 0, "wrist_roll": 0, "gripper": 3}
arm = Arm(str(HERE / "so101.urdf"), lim)
Hi = np.linalg.inv(np.array(json.load(open(setup_file("homography.json")))["H_robot_to_px"]))
cams = robot_cams(); top, wrist = open_camera(cams["top"]), open_camera(cams["wrist"])
o = fol.get_observation(); last = np.array([o[f"{m}.pos"] for m in M])


def grab(c):
    for _ in range(6): c.read()
    return c.read()[1]


def go(t, secs):
    global last
    b = np.array([t.get(m, v) for m, v in zip(M, last)]); a = last.copy()
    secs = max(secs, np.abs(b - a)[:5].max() / 30.0)
    for s in np.linspace(0, 1, max(2, int(secs * FPS))):
        s = 0.5 - 0.5 * np.cos(np.pi * s); fol.send_action({f"{m}.pos": v for m, v in zip(M, a + (b - a) * s)}); time.sleep(1 / FPS)
    last = b; time.sleep(0.3)


def cur(): return dict(zip(M, last))


def above(x, y):
    if np.hypot(x, y) < 160: raise SystemExit(f"cube too close to the base (r {np.hypot(x, y):.0f} mm); move it outward")
    q, e, ang = arm.ik_best([x, y, Z])
    if e > 5 or ang > 10: raise SystemExit(f"cannot reach above ({x:.0f}, {y:.0f})")
    go({**q, "wrist_roll": 0, "gripper": 50}, 1.5)
    return arm.fk({**q, "wrist_roll": 0})[2][:3, :3]


def cube_px():
    d = find_cube(grab(wrist), None, top=False)
    if d is None: raise SystemExit("cube not visible in the wrist camera")
    return d["c"]


try:
    for m in ["elbow_flex", "wrist_flex", "shoulder_lift"]: go({**cur(), m: mid[m]}, 1.5)
    go(HOME, 2.0)
    d = find_cube(grab(top), None, top=True)
    if d is None: raise SystemExit("cube not visible in the top camera (red face up?)")
    p = Hi @ [*d["c"], 1]; cx, cy = p[:2] / p[2]
    R = above(cx, cy); p0 = cube_px()
    shifts = []
    for dx, dy in [(STEP, 0), (0, STEP)]:
        above(cx + dx, cy + dy); shifts.append(cube_px() - p0)
    above(cx, cy)
    A = -np.array(shifts).T / STEP                     # pixel change per mm of robot x, y
    pxmm = float(np.mean(np.linalg.norm(A, axis=1)))
    axes = {}
    for name, row in (("XIMG_G", A[0]), ("YIMG_G", A[1])):
        w = np.r_[row / np.linalg.norm(row), 0.0]     # image axis in the robot xy plane
        axes[name] = (R.T @ w).round(3).tolist()       # same axis in the gripper frame
    out = {**axes, "pxmm": round(pxmm, 2)}
    json.dump(out, open(WORK / "wrist_cam.json", "w"), indent=1)
    print("saved", WORK / "wrist_cam.json", out)
finally:
    try:
        go({**cur(), "gripper": 50}, 0.5)
        go({**cur(), **{k: mid[k] for k in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]}, "wrist_roll": 0}, 2.0)
        for m in ["shoulder_lift", "wrist_flex", "elbow_flex"]: go({**cur(), m: REST[m]}, 1.6)
        go({**cur(), "gripper": 3}, 0.4)
    finally:
        for m in M:
            try: fol.bus.write("Torque_Enable", m, 0)
            except Exception: pass
        top.release(); wrist.release()
