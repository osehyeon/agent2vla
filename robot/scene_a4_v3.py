import sys, numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = np.array([144.3, -90.8]); RI, RO, MG = 190, 265, 24
EDGES = np.linspace(-40, 40, 4)
base = ["#e6eef8", "#e8f3e6", "#f8f4e2"]; zone = ["#c2d6ef", "#c6e3c1", "#efe3b0"]; edge = ["#7f9fc9", "#86b37f", "#c8b46a"]
fig = plt.figure(figsize=(297 / 25.4, 210 / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 297); ax.set_ylim(0, 210); ax.set_aspect("equal"); ax.axis("off")
yy, xx = np.mgrid[0:1050, 0:1485] / 5 + 0.1
d = np.dstack([xx - O[0], yy - O[1]]); r = np.hypot(d[..., 0], d[..., 1]); ang = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
inpaper = (xx >= MG) & (xx <= 297 - MG) & (yy >= MG) & (yy <= 210 - MG)
img = np.ones(xx.shape + (3,))
zones = []
for i in range(3):
    lo, hi = EDGES[i], EDGES[i + 1]
    sec = (ang >= lo) & (ang < hi) & (r >= RI - MG) & (r <= RO + MG)
    z = inpaper & (r >= RI) & (r <= RO) & (r * np.sin(np.radians(ang - lo)) >= MG) & (r * np.sin(np.radians(hi - ang)) >= MG)
    img[sec] = matplotlib.colors.to_rgb(base[i]); img[z] = matplotlib.colors.to_rgb(zone[i]); zones.append(z)
ax.imshow(img, extent=(0, 297, 0, 210), origin="lower", interpolation="nearest")
for i, z in enumerate(zones):
    ax.contour(xx[0], yy[:, 0], z.astype(float), levels=[0.5], colors=[edge[i]], linewidths=0.8)
th = np.radians(np.linspace(-40, 40, 400))
for rr in (RI - MG, RO + MG): ax.plot(O[0] + rr * np.sin(th), O[1] + rr * np.cos(th), color="#9a9a9a", lw=0.7)
for a in EDGES:
    t_ = np.radians(a); ax.plot([O[0] + (RI - MG) * np.sin(t_), O[0] + (RO + MG) * np.sin(t_)], [O[1] + (RI - MG) * np.cos(t_), O[1] + (RO + MG) * np.cos(t_)], color="#9a9a9a", lw=0.7)
for i in range(3):
    a = np.radians((EDGES[i] + EDGES[i + 1]) / 2); rr = RI - MG - 14
    ax.text(O[0] + rr * np.sin(a), O[1] + rr * np.cos(a), str(i + 1), fontsize=24, color="#9a9a9a", ha="center", va="center", weight="bold")
dct = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50); s = 18
for i, (x, y) in enumerate([(6, 210 - 6 - s), (297 - 6 - s, 210 - 6 - s), (297 - 6 - s, 6), (6, 6)]):
    ax.imshow(cv2.aruco.generateImageMarker(dct, i, 200), cmap="gray", extent=(x, x + s, y, y + s), interpolation="nearest", zorder=3)
fig.savefig(sys.argv[1], dpi=int(sys.argv[2]) if len(sys.argv) > 2 else 100)
for i, z in enumerate(zones): print(f"region {i + 1}: {z.sum() / 25 / 100:.0f} cm2")
