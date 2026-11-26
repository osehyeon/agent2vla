"""Fit the top-camera homography (robot XY in mm -> top image pixels) for a new table setup.

usage: python calib_table.py
The arm is limp the whole time; hold it with your hand. You will:
  1. swing the arm out of the top camera view (the four paper markers must be visible),
  2. touch the centre of each marker with the gripper tip,
  3. fold the arm back by hand.
Writes WORK/homography.json, which collect.py and trials.py read before the repo default.
"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK, FOLLOWER_PORT, FOLLOWER_ID, robot_cams, open_camera, robot_lock
from kin import Arm
from paper_pose import detect
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

robot_lock()
fol = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)); fol.connect(calibrate=False)
M = list(fol.bus.motors)
for m in M: fol.bus.write("Torque_Enable", m, 0)
arm = Arm(str(HERE / "so101.urdf"))
cam = open_camera(robot_cams()["top"])


def grab():
    for _ in range(8): cam.read()
    return cam.read()[1]


try:
    input("Torque is off. Hold the arm, swing it out of the top camera view, then press Enter.")
    img = grab(); cv2.imwrite(str(WORK / "calib_table_top.jpg"), img)
    found = detect(img)
    if len(found) < 4: raise SystemExit(f"need all 4 paper markers in view, found {sorted(found)} (see {WORK / 'calib_table_top.jpg'})")
    px, rb = [], []
    for i in sorted(found):
        input(f"Put the gripper tip on the centre of marker {i}, keep it there, press Enter.")
        o = fol.get_observation(); tip = arm.fk({m: o[f'{m}.pos'] for m in M if m != 'gripper'})[0] * 1000
        print(f"  marker {i}: robot x {tip[0]:.0f} y {tip[1]:.0f} z {tip[2]:.0f} mm")
        if abs(tip[2]) > 15: print("  warning: the tip is not at table height; press it onto the paper")
        px.append(found[i].mean(0)); rb.append(tip[:2])
    H, _ = cv2.findHomography(np.float32(rb), np.float32(px))
    json.dump({"H_robot_to_px": H.tolist(), "markers": sorted(found)}, open(WORK / "homography.json", "w"), indent=1)
    print(f"saved {WORK / 'homography.json'}")
    input("Fold the arm back to its rest pose by hand, then press Enter.")
finally:
    cam.release()
    try: fol.bus.port_handler.closePort()
    except Exception: pass
