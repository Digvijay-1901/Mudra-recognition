import cv2
import numpy as np

# bone chains per finger (from the first knuckle to the tip) and a colour for each
FINGERS = [
    ([1, 2, 3, 4],     (255, 120, 120)),   # thumb  - coral
    ([5, 6, 7, 8],     (255, 196, 70)),    # index  - amber
    ([9, 10, 11, 12],  (80, 220, 170)),    # middle - mint
    ([13, 14, 15, 16], (90, 180, 255)),    # ring   - sky
    ([17, 18, 19, 20], (190, 130, 255)),   # pinky  - violet
]
PALM = [0, 1, 5, 9, 13, 17]
EDGE = (25, 25, 35)                        # thin dark edge keeps lines visible on any background
TIPS = {4, 8, 12, 16, 20}
JOINT_COLOUR = {i: col for chain, col in FINGERS for i in chain}


def draw(img_rgb, results):
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for hl in results.multi_hand_landmarks:
        pts = np.array([[p.x * w, p.y * h] for p in hl.landmark])
        size = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 60.0)
        t = max(2, int(size * 0.016))          # bone thickness
        r = max(3, int(size * 0.020))          # joint radius
        ip = [tuple(int(v) for v in p) for p in pts]

        # 1. soft coloured glow under the bones
        glow = np.zeros_like(out)
        for chain, col in FINGERS:
            for a, b in zip(chain[:-1], chain[1:]):
                cv2.line(glow, ip[a], ip[b], col, t * 4, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), max(3, t * 2))
        out = cv2.addWeighted(out, 1.0, glow, 0.5, 0)

        # 2. translucent palm
        palm = np.array([ip[i] for i in PALM], dtype=np.int32)
        layer = out.copy()
        cv2.fillPoly(layer, [palm], (255, 255, 255), cv2.LINE_AA)
        out = cv2.addWeighted(layer, 0.16, out, 0.84, 0)
        cv2.polylines(out, [palm], True, EDGE, 2, cv2.LINE_AA)
        cv2.polylines(out, [palm], True, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. bones: dark edge first, then the colour on top
        for chain, col in FINGERS:
            for a, b in zip(chain[:-1], chain[1:]):
                cv2.line(out, ip[a], ip[b], EDGE, t + 2, cv2.LINE_AA)
        for chain, col in FINGERS:
            for a, b in zip(chain[:-1], chain[1:]):
                cv2.line(out, ip[a], ip[b], col, t, cv2.LINE_AA)

        # 4. joints: white dots with a coloured ring, bigger at the fingertips
        for i, p in enumerate(ip):
            col = JOINT_COLOUR.get(i, (255, 255, 255))
            rad = int(r * 1.5) if i in TIPS else r
            cv2.circle(out, p, rad + 2, EDGE, -1, cv2.LINE_AA)
            cv2.circle(out, p, rad, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, p, max(2, rad - 2), col, -1 if i in TIPS else max(1, rad // 3), cv2.LINE_AA)
    return out