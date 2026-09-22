import _quiet
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
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
        return _static_hands
    else:
        if _live_hands is None:
            _live_hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
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

def _shape_features(lm_raw):
    if lm_raw is None:
        return None
    lm = lm_raw - lm_raw[0]
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

def extract_double_hand_features(image_rgb, static=True):
    hands = _get_hands(static=static)
    results = hands.process(image_rgb)

    # fallback: retry with padding if nothing detected
    if not results.multi_hand_landmarks:
        h0, w0 = image_rgb.shape[:2]
        padded = cv2.copyMakeBorder(image_rgb, h0 // 3, h0 // 3, w0 // 3, w0 // 3, cv2.BORDER_REPLICATE)
        results = hands.process(padded)
        if results.multi_hand_landmarks:
            image_rgb = padded

    if not results.multi_hand_landmarks:
        return None

    h, w, _ = image_rgb.shape
    n_detected = len(results.multi_hand_landmarks)

    raw = []
    for hand_lms in results.multi_hand_landmarks:
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in hand_lms.landmark], dtype=np.float32)
        raw.append(pts)

    handed = []
    if results.multi_handedness:
        handed = [hd.classification[0].label for hd in results.multi_handedness]

    left_raw, right_raw = None, None
    if n_detected == 2 and "Left" in handed and "Right" in handed:
        left_raw = raw[handed.index("Left")]
        right_raw = raw[handed.index("Right")]
    elif n_detected == 2:
        left_raw, right_raw = raw[0], raw[1]
    else:
        if handed and handed[0] == "Right":
            right_raw = raw[0]
        else:
            left_raw = raw[0]

    zero_shape = np.zeros(63 + 5 + 4 + 5, dtype=np.float32)
    left_shape = _shape_features(left_raw)
    right_shape = _shape_features(right_raw)

    if left_shape is None and right_shape is None:
        return None

    left_shape = left_shape if left_shape is not None else zero_shape
    right_shape = right_shape if right_shape is not None else zero_shape

    if left_raw is not None and right_raw is not None:
        wrist_dist = np.linalg.norm(left_raw[0] - right_raw[0])
        palm_ids = [0, 5, 9, 13, 17]
        left_palm = np.mean(left_raw[palm_ids], axis=0)
        right_palm = np.mean(right_raw[palm_ids], axis=0)
        palm_dist = np.linalg.norm(left_palm - right_palm)
        hand_span = (np.max(np.linalg.norm(left_raw - left_raw[0], axis=1)) +
                     np.max(np.linalg.norm(right_raw - right_raw[0], axis=1))) / 2.0
        hand_span = max(hand_span, 1e-6)
        wrist_dist_norm = wrist_dist / hand_span
        palm_dist_norm = palm_dist / hand_span
        tips = [4, 8, 12, 16, 20]
        tip_dists_norm = [np.linalg.norm(left_raw[t] - right_raw[t]) / hand_span for t in tips]
        relational = np.array([wrist_dist_norm, palm_dist_norm] + tip_dists_norm, dtype=np.float32)
        both_detected_flag = 1.0
    else:
        relational = np.zeros(7, dtype=np.float32)
        both_detected_flag = 0.0

    return np.concatenate([left_shape, right_shape, relational, [both_detected_flag]])

FEATURE_DIM = (63 + 5 + 4 + 5) * 2 + 2 + 5 + 1