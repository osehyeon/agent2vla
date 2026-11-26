import json, os, signal, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK
from cube import find_cube
WD = WORK; CK = sys.argv[1]; N = int(sys.argv[2]); PREFIX = sys.argv[3]
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
prev = Path(sys.argv[4]) if len(sys.argv) > 4 else WD / "snaps" / "top.jpg"
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
    subprocess.run([sys.executable, str(HERE / "rollout.py"), "--policy", CK, "--task", task, "--duration", "25", "--tag", tag], cwd=WD,
                   env={**__import__("os").environ, "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_OFFLINE": "1"})
    meta = json.load(open(WD / "rollouts" / tag / "meta.json"))
    new = locate(WD / "rollouts" / tag / "final_top.jpg")
    ok = new is not None and new["region"] == dst and new["inside"]
    r = {"trial": k, "src": loc["region"], "dst": dst, "kind": kind, "task": task, "after": new, "stop": meta["stop"], "success": ok}
    results.append(r); print("RESULT", json.dumps(r), flush=True)
    with open(WD / "rollouts" / f"{PREFIX}_results.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
    loc = new
print("SUMMARY", sum(r["success"] for r in results), "/", len(results), flush=True)
