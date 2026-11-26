import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK, robot_cams, open_camera
from cube import find_cube
ap = argparse.ArgumentParser(description="Repeat rollouts, each time asking for a region other than the one the cube is in.")
ap.add_argument("--policy", required=True, help="checkpoint folder or Hugging Face repo id")
ap.add_argument("--n", type=int, default=10, help="number of trials")
ap.add_argument("--prefix", default=time.strftime("t%m%d_%H%M"), help="name prefix for rollouts/ and the results file")
ap.add_argument("--start-image", default=None, help="top image to find the cube in (default: take one now)")
ap.add_argument("--display", action="store_true", help="show each rollout in Rerun")
ap.add_argument("--rerun", default=None, help="HOST:PORT of a running Rerun viewer")
A = ap.parse_args(); WD = WORK; CK, N, PREFIX = A.policy, A.n, A.prefix
H = np.array(json.load(open(HERE / "homography.json"))["H_robot_to_px"]); Hi = np.linalg.inv(H)
pp = json.load(open(WD / "collect" / "paper_pose.json")); R, t = np.array(pp["R"]), np.array(pp["t"])
O = np.array([144.3, -90.8]); EDGES = np.linspace(-40, 40, 4); COL = {1: "blue", 2: "green", 3: "yellow"}
KINDS = [("number", "Move the cube to region {n}."), ("color", "Put the cube on the {c} region."),
         ("both", "Place the cube in region {n}, the {c} one."), ("unseen", "Bring the cube to the {c} zone.")]
def locate(path):
    d = find_cube(cv2.imread(str(path)))
    if d is None: return None
    p = Hi @ [*d["c"], 1]; r = p[:2] / p[2]; q = R.T @ (r - t); dd = q - O
    rr, ang = np.hypot(*dd), np.degrees(np.arctan2(dd[0], dd[1]))
    reg = int(np.clip(np.searchsorted(EDGES, ang), 1, 3)); lo, hi = EDGES[reg - 1], EDGES[reg]
    edge = min(rr * np.sin(np.radians(ang - lo)), rr * np.sin(np.radians(hi - ang)))
    inside = 190 - 10 <= rr <= 265 + 10 and min(q[0], 297 - q[0], q[1], 210 - q[1]) > 20 and edge >= 20
    return {"region": reg, "angle": round(float(ang), 1), "r": round(float(rr)), "inside": bool(inside), "edge_mm": round(float(edge))}
if A.start_image: prev = Path(A.start_image)
else:
    cam = open_camera(robot_cams()["top"])
    for _ in range(10): cam.read()
    ok, f = cam.read(); cam.release(); (WD / "snaps").mkdir(exist_ok=True); prev = WD / "snaps" / f"{PREFIX}_start.jpg"; cv2.imwrite(str(prev), f)
loc = locate(prev); print("start", loc, flush=True)
results = []; stop_req = []
signal.signal(signal.SIGINT, lambda *_: (stop_req.append(1), print("\nCtrl+C: this trial finishes its return, then trials stop", flush=True)))
for k in range(N):
    if stop_req: break
    if loc is None or not loc["inside"]:
        print("STOP: cube not found or outside the work area", loc, flush=True); break
    dst = [r for r in (1, 2, 3) if r != loc["region"]][k % 2]
    kind, tmpl = KINDS[k % len(KINDS)]; task = tmpl.format(n=dst, c=COL[dst]); tag = f"{PREFIX}_{k:02d}"
    print(f"[{k}] cube in region {loc["region"]} -> task: {task}", flush=True)
    subprocess.run([sys.executable, str(HERE / "rollout.py"), "--policy", CK, "--task", task, "--duration", "25", "--tag", tag] + (["--display"] if A.display else []) + (["--rerun", A.rerun] if A.rerun else []), cwd=WD,
                   env={**__import__("os").environ, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_OFFLINE": "1"})
    meta = json.load(open(WD / "rollouts" / tag / "meta.json"))
    new = locate(WD / "rollouts" / tag / "final_top.jpg")
    ok = new is not None and new["region"] == dst and new["inside"]
    r = {"trial": k, "src": loc["region"], "dst": dst, "kind": kind, "task": task, "after": new, "stop": meta["stop"], "success": ok}
    results.append(r); print("RESULT", json.dumps(r), flush=True)
    with open(WD / "rollouts" / f"{PREFIX}_results.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
    loc = new
print("SUMMARY", sum(r["success"] for r in results), "/", len(results), flush=True)
