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
import json
sys.path.insert(0, str(Path(__file__).parent)); from kin import Arm, C0
from scipy.spatial import cKDTree
from config import FOLLOWER_CALIB
cal = json.load(open(FOLLOWER_CALIB))
lim = {k: (-(c["range_max"] - c["range_min"]) * 360 / 4095 / 2 * 0.97, (c["range_max"] - c["range_min"]) * 360 / 4095 / 2 * 0.97) for k, c in cal.items()}
a = Arm(str(urdf), lim)
names = {j["name"]: c for c, j in joints.items()}
def pts(link, q):
    W = fk({k: np.radians(v - C0.get(k, 0)) for k, v in q.items() if k in C0})
    v = meshes[link].vertices[::7]; return (W[link] @ np.c_[v, np.ones(len(v))].T).T[:, :3]
body = ["base_link", "shoulder_link"]; hand = ["gripper_link", "moving_jaw_so101_v1_link"]
for r in (110, 120, 130, 140, 150, 160, 180):
    row = []
    for yaw in (-40, 0, 40):
        c = np.array([r * np.cos(np.radians(yaw)), r * np.sin(np.radians(yaw))])
        best = None
        for z in (0, 30):
            q, e, ang = a.ik_best((c[0], c[1], z))
            if e > 3: row.append("IK"); best = None; break
            B = cKDTree(np.vstack([pts(l, q) for l in body])); d = B.query(np.vstack([pts(l, q) for l in hand]))[0].min() * 1000
            best = d if best is None else min(best, d)
        if best is not None: row.append(f"{best:5.0f}")
    print(f"r {r}: min hand-body distance mm (yaw -40/0/40):", row)
