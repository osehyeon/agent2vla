import cv2, numpy as np
def red_mask(img, smin):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 0] < 9) | (hsv[..., 0] > 166)) & (hsv[..., 1] > smin) & (hsv[..., 2] > 70)).astype(np.uint8) * 255
def find_cube(img, roi_poly=None, top=True):
    m = red_mask(img, 90 if top else 110)
    k = 7 if top else 25
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3) if top else (9, 9), np.uint8))
    if roi_poly is not None:
        roi = np.zeros_like(m); cv2.fillPoly(roi, [np.int32(roi_poly)], 255); m &= roi
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = (500, 6000) if top else (8000, 400000)
    best = None
    for c in cs:
        a = cv2.contourArea(c)
        if not lo < a < hi: continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if max(w, h) > 1.7 * max(1, min(w, h)): continue
        if best is None or a > best[0]: best = (a, (cx, cy), (w, h), ang)
    if best is None: return None
    a, (cx, cy), (w, h), ang = best
    return {"c": np.array([cx, cy]), "size": (w, h), "ang": ((ang + 45) % 90) - 45, "area": a}
