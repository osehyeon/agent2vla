import xml.etree.ElementTree as ET
import numpy as np

C0 = {"shoulder_pan": 5.2, "shoulder_lift": 33.8, "elbow_flex": 8.4, "wrist_flex": 3.3, "wrist_roll": 0.0}
TIP = np.array([-0.011, -0.0002, -0.1044])
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis(a, t):
    a = a / np.linalg.norm(a); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


class Arm:
    def __init__(self, urdf, lim=None):
        r = ET.parse(urdf).getroot(); self.chain = {}
        for j in r.findall("joint"):
            o = j.find("origin"); T = np.eye(4)
            T[:3, :3] = _rpy(*map(float, o.get("rpy", "0 0 0").split())); T[:3, 3] = list(map(float, o.get("xyz").split()))
            ax = j.find("axis"); self.chain[j.find("child").get("link")] = (j.get("name"), j.find("parent").get("link"), T,
                                                                          np.array(list(map(float, ax.get("xyz").split()))) if ax is not None else None)
        self.lim = lim or {}

    def fk(self, deg):
        q = {k: np.radians(deg.get(k, C0[k]) - C0[k]) for k in C0}
        W = {"base_link": np.eye(4)}
        for _ in range(8):
            for c, (n, p, T, ax) in self.chain.items():
                if p in W and c not in W:
                    R = np.eye(4)
                    if ax is not None and n in q: R[:3, :3] = _axis(ax, q[n])
                    W[c] = W[p] @ T @ R
        G = W["gripper_link"]; tip = (G @ np.r_[TIP, 1])[:3]
        d = G[:3, :3] @ TIP; d /= np.linalg.norm(d)
        return tip, d, G

    def ik(self, xyz, seed, down=True, iters=200, wdown=60):
        x = np.array([seed[k] for k in ARM[:4]], float); tgt = np.array(xyz) / 1000
        def res(v):
            tip, d, _ = self.fk(dict(zip(ARM[:4], v)))
            r = list((tip - tgt) * 1000)
            if down: r += list((d - [0, 0, -1]) * wdown)
            return np.array(r)
        for _ in range(iters):
            f = res(x); Jm = np.zeros((len(f), 4))
            for i in range(4):
                e = np.zeros(4); e[i] = 0.05; Jm[:, i] = (res(x + e) - f) / 0.05
            dx = np.linalg.lstsq(Jm.T @ Jm + 0.5 * np.eye(4), -Jm.T @ f, rcond=None)[0]
            x = x + np.clip(dx, -5, 5)
            for i, k in enumerate(ARM[:4]):
                if k in self.lim: x[i] = np.clip(x[i], *self.lim[k])
            if np.linalg.norm(dx) < 1e-3: break
        tip, d, _ = self.fk(dict(zip(ARM[:4], x)))
        return dict(zip(ARM[:4], x)), np.linalg.norm(tip * 1000 - np.array(xyz)), np.degrees(np.arccos(-d[2]))


    def ik_best(self, xyz, down=True, wdown=60):
        best = None
        for l in (-20, 10, 40, 60):
            for e in (-40, 0, 30, 60):
                seed = {"shoulder_pan": 0, "shoulder_lift": l, "elbow_flex": e, "wrist_flex": 60}
                r = self.ik(xyz, seed, down, wdown=wdown)
                score = r[1] + r[2] * wdown / 60
                if best is None or score < best[0]: best = (score, r)
        return best[1]
