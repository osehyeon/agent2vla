import cv2, numpy as np, json, sys
S = 18
MK = {0: (6, 210 - 6 - S), 1: (297 - 6 - S, 210 - 6 - S), 2: (297 - 6 - S, 6), 3: (6, 6)}
def marker_paper_corners(i):
    x, y = MK[i]; return np.array([[x, y + S], [x + S, y + S], [x + S, y], [x, y]], float)  # TL TR BR BL as printed (y up)
def detect(im):
    big = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
    p = cv2.aruco.DetectorParameters(); p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    c, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), p).detectMarkers(g)
    return {} if ids is None else {int(i): cc[0] / 2 for i, cc in zip(ids.ravel(), c)}
def _fit(P, R):
    mp, mr = P.mean(0), R.mean(0); U, s_, Vt = np.linalg.svd((R - mr).T @ (P - mp))
    if np.linalg.det(U @ Vt) < 0: U[:, -1] *= -1
    Rot = U @ Vt; t = mr - Rot @ mp
    return Rot, t, np.linalg.norm((P @ Rot.T + t) - R, axis=1), np.sqrt(((R - mr) ** 2).sum() / ((P - mp) ** 2).sum())


def paper_to_robot(im, H):
    Hi = np.linalg.inv(H); det = {i: c for i, c in detect(im).items() if i in MK}
    pts = {}
    for i, cc in det.items():
        pp = marker_paper_corners(i); rr = []
        for uv in cc:
            q = Hi @ [uv[0], uv[1], 1]; rr.append(q[:2] / q[2])
        pts[i] = (pp, np.array(rr))
    ids = sorted(pts)
    while True:
        P = np.vstack([pts[i][0] for i in ids]); R = np.vstack([pts[i][1] for i in ids])
        Rot, t, res, scale = _fit(P, R)
        per = {i: res[k * 4:(k + 1) * 4].max() for k, i in enumerate(ids)}
        if res.max() <= 6 or len(ids) <= 2: return Rot, t, ids, res.max(), scale
        ids.remove(max(per, key=per.get))


if __name__ == "__main__":
    H = np.array(json.load(open(sys.argv[2]))["H_robot_to_px"])
    Rot, t, ids, rmax, sc = paper_to_robot(cv2.imread(sys.argv[1]), H)
    print("markers", ids, "max residual mm", round(rmax, 1), "scale", round(sc, 3))
    for name, p in {"paper BL": (0, 0), "BR": (297, 0), "TL": (0, 210), "TR": (297, 210), "centre": (148.5, 105)}.items():
        r = Rot @ p + t; print(name, r.round(0), "r", round(np.hypot(*r)))
    json.dump({"R": Rot.tolist(), "t": t.tolist()}, open(sys.argv[3], "w"))
