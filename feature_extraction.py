import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

_static_hands = None
_live_hands = None

def _get_hands(static=True):
    global _static_hands, _live_hands
    if static:
        if _static_hands is None:
            _static_hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        return _static_hands
    else:
        if _live_hands is None:
            _live_hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        return _live_hands

def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos = np.dot(ba, bc) / denom
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))

def extract_single_hand_features(image_rgb, static=True):
    hands = _get_hands(static=static)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        h0, w0 = image_rgb.shape[:2]
        padded = cv2.copyMakeBorder(image_rgb, h0 // 3, h0 // 3, w0 // 3, w0 // 3, cv2.BORDER_REPLICATE)
        results = hands.process(padded)

    if not results.multi_hand_landmarks:
        return None

    lm = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark], dtype=np.float32)
    if lm.shape != (21, 3) or not np.isfinite(lm).all():
        return None

    lm = lm - lm[0]
    scale = np.max(np.linalg.norm(lm, axis=1))
    if scale < 1e-6:
        return None
    lm = lm / scale

    feats = [lm.flatten()]

    tips = [4, 8, 12, 16, 20]
    for t in tips:
        feats.append([np.linalg.norm(lm[t] - lm[0])])
    for i in range(len(tips) - 1):
        feats.append([np.linalg.norm(lm[tips[i]] - lm[tips[i + 1]])])

    angle_ids = [(2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)]
    for a, b, c in angle_ids:
        feats.append([_angle(lm[a], lm[b], lm[c])])

    return np.concatenate(feats)

FEATURE_DIM = 63 + 5 + 4 + 5