import json, os, random, signal, subprocess, sys, threading, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK, FOLLOWER_PORT, FOLLOWER_ID, robot_cams, open_camera, setup_file, robot_lock
from kin import Arm
from cube import find_cube
from paper_pose import paper_to_robot, MK
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

WD = WORK; OUT = WD / "collect"; OUT.mkdir(exist_ok=True)
MODE = sys.argv[1]; NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SPEED, FPS, PER_CHUNK, CYCLE_TIMEOUT, MAX_FAILS = 1.5, 30, 5, 120, 3
H = np.array(json.load(open(setup_file("homography.json")))["H_robot_to_px"]); Hi = np.linalg.inv(H)
CAL = OUT / "cube_calib.json"
cc = json.load(open(CAL)) if CAL.exists() else {"tgt": [495, 505], "pxmm": 7.7}
OFF_G = np.array([38.5, -5.7, -3.6]) / 1000
XIMG_G = np.array([0.991, -0.1, -0.084]); YIMG_G = np.array([-0.095, -0.994, 0.053])
ZH, ZOBS, ZG, ZP = 60, 40, 0, 4
ZONE_O, ZONE_M, RMIN, RMAX = np.array([144.3, -90.8]), 24, 190, 265
EDGES = np.linspace(-40, 40, 4); NREG = 3
RMIN_ROBOT = 165
QO = {1: (148.5, 105), 2: (0, 105), 3: (0, 0), 4: (148.5, 0)}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def discover_cams():
    from paper_pose import detect
    fixed = robot_cams()
    caps = {}
    for k, i in fixed.items():
        caps[k] = open_camera(i)
    def grab(c):
        for _ in range(4): c.read()
        ok, f = c.read(); return f if ok else None
    ft, fw = grab(caps["top"]), grab(caps["wrist"])
    if ft is None or fw is None or ft.shape[1] != 1280 or fw.shape[1] != 1280:
        raise RuntimeError(f"robot cameras {fixed} not 720p - camera numbers changed? (other cameras are not probed)")
    if len(detect(ft)) < 1: raise RuntimeError(f"camera {fixed['top']} does not see the paper markers")
    a_ = cv2.cvtColor(cv2.resize(fw, (320, 180)), cv2.COLOR_BGR2GRAY).astype(float); r0 = cur()["wrist_roll"]
    go({**cur(), "wrist_roll": r0 + 25}, 0.8); b_ = cv2.cvtColor(cv2.resize(grab(caps["wrist"]), (320, 180)), cv2.COLOR_BGR2GRAY).astype(float)
    go({**cur(), "wrist_roll": r0}, 0.8)
    for c in caps.values(): c.release()
    ch = np.abs(a_ - b_).mean()
    if ch < 5: raise RuntimeError(f"camera {fixed['wrist']} did not change when the wrist moved ({ch:.1f})")
    log(f"cameras: top={fixed['top']} wrist={fixed['wrist']} (wrist change {ch:.1f}); no other camera opened")
    return fixed["top"], fixed["wrist"]


class Cam:
    def __init__(self, idx):
        self.c = open_camera(idx)
        self.f, self.t, self.run = None, 0, True
        threading.Thread(target=self._loop, daemon=True).start()
        while self.f is None: time.sleep(0.05)
    def _loop(self):
        while self.run:
            ok, f = self.c.read()
            if ok: self.f, self.t = f, time.time()
    def get(self, fresh_after=None):
        while fresh_after is not None and self.t < fresh_after: time.sleep(0.01)
        return self.f.copy()
    def close(self): self.run = False; time.sleep(0.1); self.c.release()


robot_lock()
fol = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID)); fol.connect(calibrate=False)
cal = fol.bus.calibration; M = list(fol.bus.motors)
for m in M:
    if m != "gripper": fol.bus.write("Torque_Limit", m, 1000)
lim = {k: (-(c.range_max - c.range_min) * 360 / 4095 / 2 * 0.97, (c.range_max - c.range_min) * 360 / 4095 / 2 * 0.97) for k, c in cal.items()}
arm = Arm(str(HERE / "so101.urdf"), lim)
cams = {}
o = fol.get_observation(); last = np.array([o[f"{m}.pos"] for m in M])
mid = {m: (2047 - (c.range_min + c.range_max) / 2) * 360 / 4095 for m, c in cal.items() if m != "gripper"}
HOME = {**mid, "shoulder_pan": -85, "wrist_flex": mid["wrist_flex"] + 60, "wrist_roll": 0, "gripper": 3}
REST = {**mid, "shoulder_lift": -64, "elbow_flex": 65, "wrist_flex": 0, "wrist_roll": 0, "gripper": 3}

ds = None; recording = False; task = None


def frame_out():
    st = fol.get_observation()
    imgs = {n: cv2.cvtColor(cv2.resize(c.get(), (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for n, c in cams.items()}
    return np.array([st[f"{m}.pos"] for m in M], np.float32), imgs


THR = {"shoulder_pan": 10, "shoulder_lift": 10, "elbow_flex": 10, "wrist_flex": 10, "wrist_roll": 12}
maxerr = {m: 0.0 for m in THR}
class Contact(Exception): pass


def go(t, secs, carry=False):
    global last
    secs = secs / (1.0 if carry else SPEED)
    b = np.array([t.get(m, v) for m, v in zip(M, last)]); a = last.copy()
    secs = max(secs, np.abs(b - a)[:5].max() / 40.0)
    n = max(2, int(secs * FPS)); t0 = time.time(); streak = 0
    vel = np.zeros(len(M)); prev = a.copy()
    for i, s in enumerate(np.linspace(0, 1, n)):
        s = 0.5 - 0.5 * np.cos(np.pi * s); act = a + (b - a) * s
        vel = np.abs(act - prev) * FPS; prev = act
        fol.send_action({f"{m}.pos": v for m, v in zip(M, act)})
        if recording:
            stt, imgs = frame_out()
        else:
            o_ = fol.get_observation(); stt = np.array([o_[f"{m}.pos"] for m in M])
        hit = []
        for j, m in enumerate(M):
            if m in THR:
                e_ = abs(act[j] - stt[j]); maxerr[m] = max(maxerr[m], e_)
                if e_ > THR[m] + 0.15 * vel[j]: hit.append(m)
        streak = streak + 1 if hit else 0
        if streak >= 2:
            fol.send_action({f"{m}.pos": v for m, v in zip(M, stt)}); last = np.array(stt, float)
            raise Contact(f"contact {hit} at {dict(zip(M, np.round(stt, 1).tolist()))}")
        if recording:
            ds.add_frame({"observation.state": stt, "action": act.astype(np.float32),
                          "observation.images.top": imgs["top"], "observation.images.wrist": imgs["wrist"], "task": task})
        dt = t0 + (i + 1) / FPS - time.time()
        if dt > 0: time.sleep(dt)
    last = b


def hold(secs):
    go(dict(zip(M, last)), secs, carry=True)


def cur(): return dict(zip(M, last))
def px2rb(u, v): p = Hi @ [u, v, 1]; return p[:2] / p[2]
def rb2px(x, y): p = H @ [x, y, 1]; return p[:2] / p[2]


seed = None
def tip_q(tip, strict=True):
    global seed
    if seed is not None:
        q, e, ang = arm.ik(tip, seed, iters=60)
        if e <= 3 and ang <= 8: seed = q; return q, e, ang
    q, e, ang = arm.ik_best(tip)
    if e > 5 or ang > 10:
        q2, e2, a2 = arm.ik_best(tip, wdown=8)
        if e2 <= 4 and a2 <= 25: q, e, ang = q2, e2, a2
    if strict and (e > 5 or ang > 25): raise RuntimeError(f"IK fail {np.round(tip)} err {e:.1f} tilt {ang:.1f}")
    seed = q; return q, e, ang


def solve_box(c, roll):
    tip = np.array(c, float)
    for _ in range(3):
        q, e, ang = tip_q(tip); _, _, G = arm.fk({**q, "wrist_roll": roll}); tip = np.array(c) - G[:3, :3] @ OFF_G * 1000
    if np.hypot(tip[0], tip[1]) < 150: raise RuntimeError(f"too close to body (tip r {np.hypot(tip[0], tip[1]):.0f})")
    if q["shoulder_pan"] > 55 and np.hypot(tip[0], tip[1]) < 220: raise RuntimeError("stand-side keep-out")
    return q, G


def box_pose(c, roll, grip, secs, n=1, p0=None, carry=False):
    qs = []
    for i in range(1, n + 1):
        ci = np.array(c, float) if p0 is None else np.array(p0) + (np.array(c) - np.array(p0)) * i / n
        q, G = solve_box(ci, roll); qs.append(q)
    for q in qs: go({**q, "wrist_roll": roll, "gripper": grip}, secs / n, carry)
    return G


def feasible(c, roll):
    global seed
    s0 = seed
    try: solve_box(c, roll); return True
    except RuntimeError: return False
    finally: seed = s0


def roll_for(edge, q):
    best = None
    for r in np.arange(-90, 90.5, 1.0):
        _, _, G = arm.fk({**q, "wrist_roll": r}); v = G[:3, :3] @ XIMG_G; v = v[:2] / np.linalg.norm(v[:2])
        s = min(abs(v @ edge), abs(v @ np.array([-edge[1], edge[0]])))
        if best is None or s < best[0] - 1e-3: best = (s, r)
    return best[1]


def to_home():
    global seed
    go({**cur(), "wrist_roll": 0, "gripper": 3}, 0.8)
    q = cur()
    if q["elbow_flex"] > mid["elbow_flex"] + 15 or q["shoulder_lift"] > mid["shoulder_lift"] + 15:
        go({**cur(), "shoulder_lift": mid["shoulder_lift"], "elbow_flex": mid["elbow_flex"], "wrist_flex": mid["wrist_flex"] + 60}, 1.5)
    go(HOME, 1.5); seed = None


class Paper:
    def __init__(self):
        R, t, ids, rmax, sc = paper_to_robot(cams["top"].get(time.time()), H)
        log(f"paper markers {ids} resid {rmax:.1f}mm scale {sc:.3f}")
        if len(ids) < 2 or rmax > 6 or abs(sc - 1) > 0.04: raise RuntimeError(f"paper pose bad: ids {ids} resid {rmax:.1f} scale {sc:.3f}")
        self.R, self.t = R, t
        json.dump({"R": R.tolist(), "t": t.tolist(), "ids": ids, "time": time.time()}, open(OUT / "paper_pose.json", "w"))
    def to_robot(self, p): return self.R @ np.asarray(p, float) + self.t
    def to_paper(self, r): return self.R.T @ (np.asarray(r, float) - self.t)
    def _polar(self, p):
        d = np.asarray(p, float) - ZONE_O; return np.hypot(*d), np.degrees(np.arctan2(d[0], d[1]))
    def quadrant(self, r):
        _, ang = self._polar(self.to_paper(r))
        return int(np.clip(np.searchsorted(EDGES, ang), 1, NREG))
    def in_zone(self, p, q=None):
        x, y = p; rr, ang = self._polar(p)
        if min(x, 297 - x, y, 210 - y) < ZONE_M or not RMIN <= rr <= RMAX: return False
        i = int(np.clip(np.searchsorted(EDGES, ang), 1, NREG)) - 1
        if q is not None and i != q - 1: return False
        lo, hi = EDGES[i], EDGES[i + 1]
        return rr * np.sin(np.radians(ang - lo)) >= ZONE_M and rr * np.sin(np.radians(hi - ang)) >= ZONE_M
    def roi_px(self):
        return np.array([rb2px(*self.to_robot(p)) for p in [(0, 0), (297, 0), (297, 210), (0, 210)]])
    def sample(self, q, avoid=None):
        for _ in range(800):
            p = (random.uniform(0, 297), random.uniform(0, 210))
            if not self.in_zone(p, q): continue
            r = self.to_robot(p)
            if not 205 <= np.hypot(*r) <= 260: continue
            if avoid is not None and np.linalg.norm(r - avoid) < 45: continue
            return r
        return None


def detect_top(paper):
    f = cams["top"].get(time.time() + 0.05)
    d = find_cube(f, paper.roi_px(), top=True)
    if d is None: return None
    c = px2rb(*d["c"]); a = np.radians(d["ang"]); u, v = d["c"]
    e = px2rb(u + 40 * np.cos(a), v + 40 * np.sin(a)) - c
    return c, e / np.linalg.norm(e), d


def align(c, roll, k, tag="a"):
    cx, cy = c; good = None
    for it in range(7):
        f = cams["wrist"].get(time.time() + 0.1); cv2.imwrite(str(OUT / f"{tag}_w{it}.jpg"), f)
        d = find_cube(f, None, top=False)
        if d is None:
            if good is not None and good[3] < 50:
                log("   cube lost, back to last good pose"); (cx, cy, roll, _) = good; box_pose((cx, cy, ZOBS), roll, 50, 0.5); return (cx, cy), roll, it
            raise RuntimeError("cube not in wrist view")
        u, v = d["c"]; lean = d["ang"]
        dx, dy = u - cc["tgt"][0], v - cc["tgt"][1]; err = max(abs(dx), abs(dy))
        if good is None or err < good[3]: good = (cx, cy, roll, err)
        log(f"   align {it}: cube ({u:.0f},{v:.0f}) tgt ({cc['tgt'][0]},{cc['tgt'][1]}) lean {lean:+.1f}")
        if abs(lean) >= 6:
            roll = roll - 0.6 * lean
            if abs(roll) > ROLL_SAFE: roll = wrap90(roll)
            box_pose((cx, cy, ZOBS), roll, 50, 0.5); continue
        if err < 25: return (cx, cy), roll, it
        _, _, G = arm.fk({**seed, "wrist_roll": roll})
        xw = G[:3, :3] @ XIMG_G; xw = xw[:2] / np.linalg.norm(xw[:2]); yw = G[:3, :3] @ YIMG_G; yw = yw[:2] / np.linalg.norm(yw[:2])
        cx, cy = np.array([cx, cy]) - 0.8 * (xw * dx + yw * dy) / cc["pxmm"]
        box_pose((cx, cy, ZOBS), roll, 50, 0.5)
    if good is not None and good[3] < 50:
        (cx, cy, roll, _) = good; box_pose((cx, cy, ZOBS), roll, 50, 0.5); return (cx, cy), roll, it
    raise RuntimeError("alignment did not converge")


ROLL_SAFE = 60
def wrap90(r): return ((r + 45) % 90) - 45


def choose_roll(c, base, heights=(ZOBS, ZG)):
    lo, hi = -ROLL_SAFE, ROLL_SAFE
    for d in (0, 90, -90):
        r = wrap90(base) + d
        if not lo <= r <= hi: continue
        if all(feasible((c[0], c[1], z), r) for z in heights): return r
    raise RuntimeError(f"no roll reaches {np.round(c)}")


def pickable(c, base):
    c = np.asarray(c, float)
    for cc_ in (c, c * (1 - 15 / np.hypot(*c))):
        ok = False
        for d in (0, 90, -90):
            r_ = wrap90(base) + d
            if abs(r_) <= ROLL_SAFE and all(feasible((cc_[0], cc_[1], z), r_) for z in (ZH, ZOBS, ZG)): ok = True; break
        if not ok: return False
    return True


class Timeout(Exception): pass
def _alarm(*_): raise Timeout("timeout")
signal.signal(signal.SIGALRM, _alarm)


def pick_place(paper, src_r, edge, dst, k, r):
    q0, _, _ = tip_q(np.array([src_r[0], src_r[1], ZOBS]), strict=False)
    try: roll = choose_roll(src_r, roll_for(edge, q0))
    except RuntimeError:
        src_r = np.asarray(src_r) * (1 + 15 / np.hypot(*src_r)); r["shifted_out"] = True
        roll = choose_roll(src_r, roll_for(edge, q0))
    zh = next((z for z in (ZH, 50, 45) if feasible((src_r[0], src_r[1], z), roll)), None)
    if zh is None: raise RuntimeError("pick approach unreachable")
    box_pose((src_r[0], src_r[1], zh), roll, 50, 1.6)
    box_pose((src_r[0], src_r[1], ZOBS), roll, 50, 0.5)
    (cx, cy), roll, it = align(src_r, roll, k); r["align_iters"] = it
    box_pose((cx, cy, ZG), roll, 50, 1.8, n=3, p0=(cx, cy, ZOBS), carry=True)
    go({**cur(), "gripper": 0}, 0.5); hold(0.25)
    g = fol.get_observation()["gripper.pos"]; r["grip"] = round(g, 1)
    if g < 5:
        go({**cur(), "gripper": 50}, 0.4); box_pose((cx, cy, ZOBS), roll, 50, 0.6)
        raise RuntimeError(f"missed (gripper {g:.1f})")
    HOLD = max(g - 6, 0); go({**cur(), "gripper": HOLD}, 0.15, carry=True)
    state = {"holding": (cx, cy), "roll": roll, "HOLD": HOLD}
    try:
        zl = next((z for z in (ZH, 50, 45) if feasible((cx, cy, z), roll)), ZOBS)
        box_pose((cx, cy, zl), roll, HOLD, 0.8, n=2, p0=(cx, cy, ZG), carry=True)
        if fol.get_observation()["gripper.pos"] < 3: raise RuntimeError("dropped on lift")
        tgt = None; lo, hi = lim["wrist_roll"]
        for _ in range(30):
            cand = paper.sample(dst)
            if cand is None: continue
            if not pickable(cand, roll): continue
            for d in (0, 90, -90):
                ro = wrap90(roll) + d
                if -ROLL_SAFE <= ro <= ROLL_SAFE and feasible((cand[0], cand[1], ZH), ro) and feasible((cand[0], cand[1], ZP), ro): tgt, proll = cand, ro; break
            if tgt is not None: break
        if tgt is None: raise RuntimeError(f"q{dst} no reachable spot")
        r["dst_xy"] = np.round(tgt, 1).tolist()
        a0, a1 = np.arctan2(cy, cx), np.arctan2(tgt[1], tgt[0]); r0, r1 = np.hypot(cx, cy), np.hypot(*tgt)
        n_w = 6
        for i in range(1, n_w + 1):
            f_ = i / n_w; ang = a0 + (a1 - a0) * f_; rad = max(r0 + (r1 - r0) * f_, RMIN_ROBOT)
            z_ = zl + (ZH - zl) * f_; ro = roll + (proll - roll) * f_
            box_pose((rad * np.cos(ang), rad * np.sin(ang), z_), ro, HOLD, 1.8 / n_w, carry=True)
        roll = proll
        box_pose((tgt[0], tgt[1], ZP), roll, HOLD, 0.8, n=2, p0=(tgt[0], tgt[1], ZH), carry=True)
        if fol.get_observation()["gripper.pos"] < 3: raise RuntimeError("dropped in transit")
        go({**cur(), "gripper": 50}, 0.4); state["holding"] = None
        box_pose((tgt[0], tgt[1], ZH), roll, 50, 0.6)
    except Exception:
        if state["holding"] is not None:
            try:
                box_pose((cx, cy, ZP), roll, HOLD, 1.0, carry=True); go({**cur(), "gripper": 50}, 0.4); box_pose((cx, cy, ZOBS), roll, 50, 0.5)
            except Exception: go({**cur(), "gripper": 50}, 0.4)
        raise
    return tgt


def new_chunk():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    i = 0
    while (OUT / "datasets" / f"cube_{i:03d}").exists(): i += 1
    root = OUT / "datasets" / f"cube_{i:03d}"
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": M},
             "action": {"dtype": "float32", "shape": (6,), "names": M},
             "observation.images.top": {"dtype": "video", "shape": (360, 640, 3), "names": ["height", "width", "channels"]},
             "observation.images.wrist": {"dtype": "video", "shape": (360, 640, 3), "names": ["height", "width", "channels"]}}
    d = LeRobotDataset.create(repo_id=f"local/cube_{i:03d}", fps=FPS, features=feats, root=root, robot_type="so101_follower",
                              use_videos=True, image_writer_threads=4)
    log(f"chunk {root.name} opened"); return d


def close_chunk():
    global ds
    if ds is not None:
        try:
            ds.finalize(); n_ = ds.meta.total_episodes; log(f"chunk saved: {ds.root.name} episodes {n_}")
            if n_ == 0:
                import shutil; shutil.rmtree(ds.root, ignore_errors=True); log("  empty chunk removed")
        except Exception as ex: log(f"finalize error {ex!r}")
        ds = None


def shutdown():
    global recording
    recording = False
    signal.alarm(25)
    try:
        for t in [{"gripper": 50}, {"wrist_roll": 0}]:
            try: go({**cur(), **t}, 0.6)
            except Exception: pass
        go({**cur(), **{k: mid[k] for k in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]}}, 2.0)
        for m in ["shoulder_lift", "wrist_flex", "elbow_flex"]: go({**cur(), m: REST[m]}, 1.6)
        log("maxerr", {m: round(v, 1) for m, v in maxerr.items()})
        go({**cur(), "gripper": 3}, 0.4)
    except BaseException as ex: log(f"shutdown motion: {ex!r}")
    signal.alarm(0)
    bad = []
    for m in M:
        for _ in range(5):
            try: fol.bus.write("Torque_Enable", m, 0); break
            except Exception: time.sleep(0.3)
        else: bad.append(m)
    try: fol.bus.port_handler.closePort()
    except Exception: pass
    for c in cams.values(): c.close()
    return bad


stop = "done"; stats = {"ok": 0, "fail": 0}
try:
    for m in ["elbow_flex", "wrist_flex", "shoulder_lift"]: go({**cur(), m: mid[m]}, 1.5)
    ti, wi = discover_cams(); cams.update({"top": Cam(ti), "wrist": Cam(wi)})
    to_home()
    paper = Paper() if MODE != "home" else None
    if MODE == "home":
        hold(1.0); cv2.imwrite(str(OUT / "home_top.jpg"), cams["top"].get(time.time() + 0.1))
    elif MODE == "check":
        cv2.imwrite(str(OUT / "home_top.jpg"), cams["top"].get(time.time() + 0.1))
        det = detect_top(paper)
        log("cube", None if det is None else (det[0].round(0), paper.quadrant(det[0])))
        for q in (3, 4):
            x0, y0 = QO[q]; near = paper.to_robot((x0 + (122.5 if q == 3 else 26), y0 + 26))
            log(f"near corner q{q} robot {near.round(0)} r {np.hypot(*near):.0f}")
            ro = choose_roll(near, 0, (50, 30)); log(f"  roll {ro}")
            box_pose((near[0], near[1], 50), ro, 50, 2.5); box_pose((near[0], near[1], 30), ro, 50, 2.0)
            cv2.imwrite(str(OUT / f"near_q{q}_top.jpg"), cams["top"].get(time.time())); hold(0.5); to_home()
    elif MODE == "calib":
        CANDS = [(495, 560), (495, 600), (495, 520), (495, 640), (450, 580), (540, 580), (450, 540), (540, 540), (495, 680)]
        ok = False
        for n_try, tg in enumerate(CANDS):
            det = detect_top(paper)
            if det is None: raise RuntimeError("cube not visible from top (red face down?) - please reset the cube")
            c, e, d = det
            if np.hypot(*c) < RMIN_ROBOT: raise RuntimeError(f"cube too close to base (r {np.hypot(*c):.0f})")
            q0, _, _ = tip_q(np.array([c[0], c[1], ZOBS]), strict=False); roll = choose_roll(c, roll_for(e, q0))
            cc["tgt"] = list(tg)
            box_pose((c[0], c[1], ZH), roll, 50, 1.6); box_pose((c[0], c[1], ZOBS), roll, 50, 0.6)
            (cx, cy), roll, it = align(c, roll, 0, tag=f"cal{n_try}")
            box_pose((cx, cy, ZG), roll, 50, 1.8, n=3, p0=(cx, cy, ZOBS), carry=True)
            cv2.imwrite(str(OUT / f"cal{n_try}_down.jpg"), cams["wrist"].get(time.time() + 0.08))
            go({**cur(), "gripper": 0}, 0.5); hold(0.3); g = fol.get_observation()["gripper.pos"]
            log(f"TRY {n_try} tgt {tg}: gripper {g:.1f}")
            if g >= 5:
                go({**cur(), "gripper": 50}, 0.5); box_pose((cx, cy, ZOBS), roll, 50, 1.2, n=2, p0=(cx, cy, ZG), carry=True); hold(0.4)
                f = cams["wrist"].get(time.time() + 0.1); cv2.imwrite(str(OUT / "calib_after.jpg"), f); d1 = find_cube(f, None, top=False)
                if d1 is None: raise RuntimeError("cube not visible after release")
                cc["tgt"] = [int(round(d1["c"][0])), int(round(d1["c"][1]))]; cc["grip_closed"] = round(g, 1); cc["found_with"] = list(tg)
                json.dump(cc, open(CAL, "w")); log("CALIB SAVED", cc); ok = True; break
            go({**cur(), "gripper": 50}, 0.4); box_pose((cx, cy, ZH), roll, 50, 1.0, n=2, p0=(cx, cy, ZG))
            to_home(); hold(0.3)
        if not ok: raise RuntimeError("no successful grasp in all candidates")
    elif MODE == "collect":
        random.seed(int(time.time())); fails = 0; k = 0
        while k < NEP:
            if ds is None: ds = new_chunk()
            r = {"ep_try": k, "t": time.strftime("%H:%M:%S")}; t0 = time.time()
            signal.alarm(CYCLE_TIMEOUT)
            try:
                det = detect_top(paper)
                if det is None: raise RuntimeError("no cube in top view")
                src_r, edge, d = det; src = paper.quadrant(src_r)
                if np.hypot(*src_r) < 170: raise RuntimeError(f"cube too close to base (r {np.hypot(*src_r):.0f})")
                dst = random.choice([q for q in range(1, NREG + 1) if q != src]); r.update(src=src, dst=dst)
                task = f"Move the cube to region {dst}."
                recording = True
                pick_place(paper, src_r, edge, dst, k, r)
                to_home(); hold(0.3); recording = False
                after = detect_top(paper)
                if after is None or paper.quadrant(after[0]) != dst:
                    raise RuntimeError(f"verify failed: {'not seen' if after is None else 'q' + str(paper.quadrant(after[0]))}")
                signal.alarm(0)
                ds.save_episode(parallel_encoding=False); stats["ok"] += 1; fails = 0; k += 1; r["result"] = "ok"
                if ds.meta.total_episodes >= PER_CHUNK: close_chunk()
            except BaseException as ex:
                signal.alarm(0); recording = False
                if isinstance(ex, KeyboardInterrupt): raise
                r["result"] = f"FAIL {ex}"; stats["fail"] += 1; fails += 1
                try: ds.clear_episode_buffer()
                except Exception: pass
                signal.alarm(20)
                try: to_home()
                except BaseException as ex3: r["result"] += f" (home err {ex3!r})"
                signal.alarm(0)
            r["secs"] = round(time.time() - t0, 1); log("EP", json.dumps(r))
            with open(OUT / "log.jsonl", "a") as fp: fp.write(json.dumps(r) + "\n")
            if fails >= MAX_FAILS: stop = f"{fails} failures in a row"; break
except BaseException as ex:
    stop = f"error: {ex!r}"
finally:
    signal.alarm(0); recording = False
    close_chunk()
    bad = shutdown()
    log(f"SUMMARY mode {MODE} ok {stats['ok']} fail {stats['fail']} stop: {stop} torque-off failed: {bad or 'none'}")
    log("END")
