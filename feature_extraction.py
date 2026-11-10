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

def single_hand_features_from_landmarks(hand_lms, image_shape=None):
    lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float32)
    if image_shape is not None:
        h, w = image_shape[:2]
        lm = lm * np.array([w, h, w], dtype=np.float32)
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


def extract_single_hand_features(image_rgb, static=True):
    hands = _get_hands(static=static)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        h0, w0 = image_rgb.shape[:2]
        padded = cv2.copyMakeBorder(image_rgb, h0 // 3, h0 // 3, w0 // 3, w0 // 3, cv2.BORDER_REPLICATE)
        results = hands.process(padded)

    if not results.multi_hand_landmarks:
        return None

    return single_hand_features_from_landmarks(results.multi_hand_landmarks[0])
FEATURE_DIM = 63 + 5 + 4 + 5

def hand_axis(hands, image_shape, *, two_wrists=False, handedness=()):
    h, w = image_shape[:2]
    scale = np.array([w, h])
    if two_wrists and len(hands) == 2:
        if len(handedness) == 2 and set(handedness) == {"Left", "Right"}:
            left, right = (hands[handedness.index(label)] for label in ("Left", "Right"))
            axis = (right[0, :2] - left[0, :2]) * scale
            if np.linalg.norm(axis) >= 1e-6:
                return axis, 0.0
        else:
            a, b = hands
            axis = (b[0, :2] - a[0, :2]) * scale
            palms = ((a[9, :2] - a[0, :2]) + (b[9, :2] - b[0, :2])) * scale
            direction = np.dot(axis, palms)
            if abs(direction) >= 1e-6:
                return axis * np.sign(direction), 0.0
    hand = hands[0]
    return (hand[9, :2] - hand[0, :2]) * scale, -90.0


def axis_angle(axis):
    if np.linalg.norm(axis) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def rotate_landmark_arrays(hands, image_shape, degrees):
    if degrees == 0:
        return hands
    h, w = image_shape[:2]
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    points = [np.stack([hand[:, 0] * w, hand[:, 1] * h], axis=1) for hand in hands]
    center = np.mean([p[0] for p in points], axis=0)
    rotated = []
    for hand, p in zip(hands, points):
        d = p - center
        q = hand.copy()
        q[:, 0] = (d[:, 0] * c - d[:, 1] * s + center[0]) / w
        q[:, 1] = (d[:, 0] * s + d[:, 1] * c + center[1]) / h
        rotated.append(q)
    return rotated


def upright_landmark_arrays(hands, image_shape, *, two_wrists=False, handedness=(), jitter=0):
    axis, target = hand_axis(hands, image_shape, two_wrists=two_wrists, handedness=handedness)
    angle = axis_angle(axis)
    correction = target - angle if angle is not None else 0.0
    return rotate_landmark_arrays(hands, image_shape, correction + jitter)


def _hand_array(hand_lms):
    return np.asarray([(p.x, p.y, p.z) for p in hand_lms.landmark], dtype=np.float32)


def hand_tilt_degrees(hand_lms, image_shape):
    axis, target = hand_axis([_hand_array(hand_lms)], image_shape)
    angle = axis_angle(axis)
    return None if angle is None else (angle - target + 180) % 360 - 180


def normalized_single_hand_features(hand_lms, image_shape):
    from types import SimpleNamespace

    hand = upright_landmark_arrays([_hand_array(hand_lms)], image_shape)[0]
    canonical = SimpleNamespace(landmark=[SimpleNamespace(x=float(x), y=float(y), z=float(z))
                                          for x, y, z in hand])
    return single_hand_features_from_landmarks(canonical, image_shape)
