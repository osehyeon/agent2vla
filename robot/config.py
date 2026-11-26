import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("AGENT2VLA_WORK", Path.home() / "lerobot-work"))
FOLLOWER_PORT = os.environ.get("AGENT2VLA_FOLLOWER_PORT", "/dev/ttyACM0")
FOLLOWER_ID = os.environ.get("AGENT2VLA_FOLLOWER_ID", "my_follower")
CALIB_DIR = Path(os.environ.get("HF_LEROBOT_CALIBRATION", Path.home() / ".cache/huggingface/lerobot/calibration"))
FOLLOWER_CALIB = CALIB_DIR / "robots" / "so_follower" / f"{FOLLOWER_ID}.json"


def robot_cams():
    """Camera ids for 'top' and 'wrist': a machine copy in WORK overrides the repo default."""
    import json
    for p in (WORK / "robot_cams.json", HERE / "robot_cams.json"):
        if p.exists(): return json.load(open(p))
    raise FileNotFoundError("robot_cams.json")


def open_camera(src, width=1280, height=720):
    """AVFoundation index on macOS, V4L2 device path or index (MJPG) on Linux."""
    import sys, cv2
    if sys.platform == "darwin":
        c = cv2.VideoCapture(int(src), cv2.CAP_AVFOUNDATION)
    else:
        c = cv2.VideoCapture(src, cv2.CAP_V4L2); c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, width); c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return c
