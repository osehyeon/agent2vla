import sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trimesh.transformations import euler_matrix, translation_matrix, rotation_matrix

urdf, out = Path(sys.argv[1]), Path(sys.argv[2])
for f in font_manager.findSystemFonts():
    if "AppleSDGothicNeo" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
root = ET.parse(urdf).getroot()


def T(o):
    xyz = [float(x) for x in o.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
    return translation_matrix(xyz) @ euler_matrix(*rpy, "sxyz")


joints = {}
for j in root.findall("joint"):
    lim = j.find("limit")
    joints[j.find("child").get("link")] = dict(name=j.get("name"), parent=j.find("parent").get("link"), T=T(j.find("origin")),
                                               axis=np.array([float(x) for x in (j.find("axis").get("xyz") if j.find("axis") is not None else "0 0 1").split()]),
                                               lo=float(lim.get("lower")) if lim is not None else 0, hi=float(lim.get("upper")) if lim is not None else 0)
meshes = {}
for link in root.findall("link"):
    ms = []
    for vis in link.findall("visual"):
        m = trimesh.load(urdf.parent / vis.find("geometry/mesh").get("filename"))
        m.apply_transform(T(vis.find("origin"))); ms.append(m)
    if ms:
        meshes[link.get("name")] = trimesh.util.concatenate(ms)


def fk(q):
    W = {"base_link": np.eye(4)}
    for _ in range(10):
        for c, j in joints.items():
            if j["parent"] in W and c not in W:
                R = rotation_matrix(q.get(j["name"], 0.0), j["axis"]) if np.any(j["axis"]) else np.eye(4)
                W[c] = W[j["parent"]] @ j["T"] @ R
    return W


def downstream(link):
    out, frontier = {link}, [link]
    while frontier:
        p = frontier.pop()
        for c, j in joints.items():
            if j["parent"] == p and c not in out:
                out.add(c); frontier.append(c)
    return out
L = np.array([0.3, -0.6, 0.75]); L /= np.linalg.norm(L)


def draw(ax, q, links, col, alpha):
    W = fk(q)
    for ln in links:
        if ln not in meshes:
            continue
        m = meshes[ln].copy(); m.apply_transform(W[ln])
        s = 0.4 + 0.6 * np.abs(m.face_normals @ L)
        fc = np.c_[np.outer(s, col), np.full(len(s), alpha)]
        ax.add_collection3d(Poly3DCollection(m.vertices[m.faces][::1], facecolors=fc, edgecolor="none"))


import json
sys.path.insert(0, str(Path(__file__).parent)); from kin import C0
from config import FOLLOWER_CALIB
cal = json.load(open(FOLLOWER_CALIB))
mid = {m: (2047 - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095 for m, c in cal.items() if m != "gripper"}
POLE = np.array([[-25, -155, -20], [25, -105, 480]]) / 1000.0
def hand_pts(q):
    W = fk({k: np.radians(v - C0[k]) for k, v in q.items() if k in C0})
    P = []
    for l in ["wrist_link", "gripper_link", "moving_jaw_so101_v1_link"]:
        v = meshes[l].vertices[::5]; P.append((W[l] @ np.c_[v, np.ones(len(v))].T).T[:, :3])
    return np.vstack(P)
def box_dist(P, B):
    d = np.maximum(np.maximum(B[0] - P, P - B[1]), 0); return np.linalg.norm(d, axis=1).min() * 1000
poses = {
 "충돌③ 대기(+85, 팔 세움)": {**mid, "shoulder_pan": 85, "shoulder_lift": mid["shoulder_lift"] - 20, "elbow_flex": mid["elbow_flex"] - 70, "wrist_roll": 0},
 "이전 대기(+85, 수평)": {**mid, "shoulder_pan": 85, "wrist_flex": mid["wrist_flex"] + 60, "wrist_roll": 0},
 "새 대기(-85, 수평)": {**mid, "shoulder_pan": -85, "wrist_flex": mid["wrist_flex"] + 60, "wrist_roll": 0},
 "접힌 자세": {**mid, "shoulder_lift": -64, "elbow_flex": 78, "wrist_flex": 72, "wrist_roll": 0},
 "접기 중간(+85에서 접힘)": {**mid, "shoulder_pan": 85, "shoulder_lift": -64, "elbow_flex": 78, "wrist_flex": 72, "wrist_roll": 30},
}
for k, q in poses.items():
    print(f"{k:28s} 기둥까지 {box_dist(hand_pts(q), POLE):6.0f} mm")
from kin import Arm
lim = {k: (-(c["range_max"] - c["range_min"]) * 360 / 4095 / 2 * 0.97, (c["range_max"] - c["range_min"]) * 360 / 4095 / 2 * 0.97) for k, c in cal.items()}
A = Arm(str(urdf), lim); OFF = np.array([38.5, -5.7, -3.6]) / 1000
pts = [(141, 111), (156, -85), (225, 117), (246, 119), (228, 69), (250, 71), (236, -31), (257, -29), (240, -79), (261, -77)]
mins = []
for c in pts:
    for z in (60, 0):
        for roll in (0, 90, -90):
            tip = np.array([c[0], c[1], z], float); ok = True
            for _ in range(3):
                q, e, ang = A.ik_best(tip)
                if e > 5: ok = False; break
                _, _, G = A.fk({**q, "wrist_roll": roll}); tip = np.array([c[0], c[1], z]) - G[:3, :3] @ OFF * 1000
            if ok: mins.append(box_dist(hand_pts({**q, "wrist_roll": roll}), POLE)); break
print("probe 통과 지점들: 기둥까지 최소", round(min(mins)), "mm  (지점 수", len(mins), ")")
fig = plt.figure(figsize=(16, 7))
for k, (name, q) in enumerate([("충돌③ 대기(+85, 팔 세움)", poses["충돌③ 대기(+85, 팔 세움)"]), ("새 대기(-85, 수평)", poses["새 대기(-85, 수평)"])]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    Wq = {kk: np.radians(v - C0[kk]) for kk, v in q.items() if kk in C0}
    draw(ax, Wq, list(meshes), np.array([.45, .45, .5]), 1.0)
    b0, b1 = POLE; xs, ys, zs = [b0[0], b1[0]], [b0[1], b1[1]], [b0[2], b1[2]]
    for X in xs:
        for Y in ys: ax.plot([X, X], [Y, Y], zs, color="#c0392b", lw=2)
    for Z in zs:
        ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]], [Z] * 5, color="#c0392b", lw=2)
    ax.set_xlim(-0.25, 0.35); ax.set_ylim(-0.35, 0.25); ax.set_zlim(-0.05, 0.55); ax.set_box_aspect((1, 1, 1)); ax.view_init(25, -60); ax.set_axis_off()
    ax.set_title(f"{name}\n손(집게·손목)–기둥 {box_dist(hand_pts(q), POLE):.0f} mm", fontsize=15)
fig.text(0.5, 0.02, "빨강: 스탠드 기둥 (사진에서 추정, 받침 오른쪽 130 mm, 높이 480 mm)", ha="center", fontsize=13)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "sim_stand.png", dpi=70)
