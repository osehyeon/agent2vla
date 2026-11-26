import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("AGENT2VLA_WORK", Path.home() / "lerobot-work"))
FOLLOWER_PORT = os.environ.get("AGENT2VLA_FOLLOWER_PORT", "/dev/ttyACM0")
FOLLOWER_ID = os.environ.get("AGENT2VLA_FOLLOWER_ID", "my_follower")
CALIB_DIR = Path(os.environ.get("HF_LEROBOT_CALIBRATION", Path.home() / ".cache/huggingface/lerobot/calibration"))
FOLLOWER_CALIB = CALIB_DIR / "robots" / "so_follower" / f"{FOLLOWER_ID}.json"


def setup_file(name):
    """Per-machine setup files (robot_cams.json, homography.json): a copy in WORK overrides the repo default."""
    for p in (WORK / name, HERE / name):
        if p.exists(): return p
    raise FileNotFoundError(name)


def robot_cams():
    import json
    return json.load(open(setup_file("robot_cams.json")))


def open_camera(src, width=1280, height=720):
    """AVFoundation index on macOS, V4L2 device path or index (MJPG) on Linux."""
    import sys, cv2
    if sys.platform == "darwin":
        c = cv2.VideoCapture(int(src), cv2.CAP_AVFOUNDATION)
    else:
        c = cv2.VideoCapture(src, cv2.CAP_V4L2); c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, width); c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return c


_lock = []


def robot_lock():
    """Allow one program at a time on the arm (across users). Exits if another holds it."""
    import fcntl, sys
    if os.environ.get("AGENT2VLA_LOCK_HELD"): return
    path = os.environ.get("AGENT2VLA_LOCK", "/tmp/agent2vla-robot.lock")
    old = os.umask(0)
    try: fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    finally: os.umask(old)
    try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: sys.exit("The robot is in use by another program. Wait for it to finish.")
    _lock.append(fd); os.environ["AGENT2VLA_LOCK_HELD"] = "1"
