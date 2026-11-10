"""Frozen pure geometry from current production; no MediaPipe dependency. Z=0 input is explicit."""

import numpy as np

def _angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-06:
        return 0.0
    cos = np.dot(ba, bc) / denom
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))

def single_hand_features_from_landmarks(hand_lms, image_shape=None):
    lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float32)
    if image_shape is not None:
        (h, w) = image_shape[:2]
        lm = lm * np.array([w, h, w], dtype=np.float32)
    if lm.shape != (21, 3) or not np.isfinite(lm).all():
        return None
    lm = lm - lm[0]
    scale = np.max(np.linalg.norm(lm, axis=1))
    if scale < 1e-06:
        return None
    lm = lm / scale
    feats = [lm.flatten()]
    tips = [4, 8, 12, 16, 20]
    for t in tips:
        feats.append([np.linalg.norm(lm[t] - lm[0])])
    for i in range(len(tips) - 1):
        feats.append([np.linalg.norm(lm[tips[i]] - lm[tips[i + 1]])])
    angle_ids = [(2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)]
    for (a, b, c) in angle_ids:
        feats.append([_angle(lm[a], lm[b], lm[c])])
    return np.concatenate(feats)

def hand_axis(hands, image_shape, *, two_wrists=False, handedness=()):
    (h, w) = image_shape[:2]
    scale = np.array([w, h])
    if two_wrists and len(hands) == 2:
        if len(handedness) == 2 and set(handedness) == {'Left', 'Right'}:
            (left, right) = (hands[handedness.index(label)] for label in ('Left', 'Right'))
            axis = (right[0, :2] - left[0, :2]) * scale
            if np.linalg.norm(axis) >= 1e-06:
                return (axis, 0.0)
        else:
            (a, b) = hands
            axis = (b[0, :2] - a[0, :2]) * scale
            palms = (a[9, :2] - a[0, :2] + (b[9, :2] - b[0, :2])) * scale
            direction = np.dot(axis, palms)
            if abs(direction) >= 1e-06:
                return (axis * np.sign(direction), 0.0)
    hand = hands[0]
    return ((hand[9, :2] - hand[0, :2]) * scale, -90.0)

def axis_angle(axis):
    if np.linalg.norm(axis) < 1e-06:
        return None
    return float(np.degrees(np.arctan2(axis[1], axis[0])))

def rotate_landmark_arrays(hands, image_shape, degrees):
    if degrees == 0:
        return hands
    (h, w) = image_shape[:2]
    (c, s) = (np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees)))
    points = [np.stack([hand[:, 0] * w, hand[:, 1] * h], axis=1) for hand in hands]
    center = np.mean([p[0] for p in points], axis=0)
    rotated = []
    for (hand, p) in zip(hands, points):
        d = p - center
        q = hand.copy()
        q[:, 0] = (d[:, 0] * c - d[:, 1] * s + center[0]) / w
        q[:, 1] = (d[:, 0] * s + d[:, 1] * c + center[1]) / h
        rotated.append(q)
    return rotated

def upright_landmark_arrays(hands, image_shape, *, two_wrists=False, handedness=(), jitter=0):
    (axis, target) = hand_axis(hands, image_shape, two_wrists=two_wrists, handedness=handedness)
    angle = axis_angle(axis)
    correction = target - angle if angle is not None else 0.0
    return rotate_landmark_arrays(hands, image_shape, correction + jitter)

def _hand_array(hand_lms):
    return np.asarray([(p.x, p.y, p.z) for p in hand_lms.landmark], dtype=np.float32)

def hand_tilt_degrees(hand_lms, image_shape):
    (axis, target) = hand_axis([_hand_array(hand_lms)], image_shape)
    angle = axis_angle(axis)
    return None if angle is None else (angle - target + 180) % 360 - 180

def normalized_single_hand_features(hand_lms, image_shape):
    from types import SimpleNamespace
    hand = upright_landmark_arrays([_hand_array(hand_lms)], image_shape)[0]
    canonical = SimpleNamespace(landmark=[SimpleNamespace(x=float(x), y=float(y), z=float(z)) for (x, y, z) in hand])
    return single_hand_features_from_landmarks(canonical, image_shape)

def _shape_features(lm_raw):
    if lm_raw is None:
        return None
    lm = lm_raw - lm_raw[0]
    scale = np.max(np.linalg.norm(lm, axis=1))
    if scale < 1e-06:
        return None
    lm = lm / scale
    feats = [lm.flatten()]
    tips = [4, 8, 12, 16, 20]
    for t in tips:
        feats.append([np.linalg.norm(lm[t] - lm[0])])
    for i in range(len(tips) - 1):
        feats.append([np.linalg.norm(lm[tips[i]] - lm[tips[i + 1]])])
    angle_ids = [(2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)]
    for (a, b, c) in angle_ids:
        feats.append([_angle(lm[a], lm[b], lm[c])])
    return np.concatenate(feats)

def double_hand_features_from_results(results, image_shape):
    if not results.multi_hand_landmarks:
        return None
    (h, w) = image_shape[:2]
    n_detected = len(results.multi_hand_landmarks)
    raw = []
    for hand_lms in results.multi_hand_landmarks:
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in hand_lms.landmark], dtype=np.float32)
        raw.append(pts)
    handed = []
    if results.multi_handedness:
        handed = [hd.classification[0].label for hd in results.multi_handedness]
    (left_raw, right_raw) = (None, None)
    if n_detected == 2 and 'Left' in handed and ('Right' in handed):
        left_raw = raw[handed.index('Left')]
        right_raw = raw[handed.index('Right')]
    elif n_detected == 2:
        (left_raw, right_raw) = (raw[0], raw[1])
    elif handed and handed[0] == 'Right':
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
        hand_span = (np.max(np.linalg.norm(left_raw - left_raw[0], axis=1)) + np.max(np.linalg.norm(right_raw - right_raw[0], axis=1))) / 2.0
        hand_span = max(hand_span, 1e-06)
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

def normalized_double_hand_features(results, image_shape):
    from types import SimpleNamespace
    hands = [_hand_array(hand) for hand in results.multi_hand_landmarks or []]
    if not hands:
        return None
    handedness = [entry.classification[0].label for entry in results.multi_handedness or []]
    hands = upright_landmark_arrays(hands, image_shape, two_wrists=True, handedness=handedness)
    canonical = SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=[SimpleNamespace(x=float(x), y=float(y), z=float(z)) for (x, y, z) in hand]) for hand in hands], multi_handedness=results.multi_handedness)
    return double_hand_features_from_results(canonical, image_shape)
