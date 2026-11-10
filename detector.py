from threading import Lock

import cv2
from feature_extraction_double import _get_hands

_hands_lock = Lock()


def detect(image_rgb, *, retry_reflect=True):
    # Streamlit sessions share this native graph. Serialize its initialization
    # and processing; static mode keeps tracking out of independent photos.
    with _hands_lock:
        return _detect(image_rgb, retry_reflect=retry_reflect)


def _detect(image_rgb, *, retry_reflect=True):
    hands = _get_hands(static=True)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        h0, w0 = image_rgb.shape[:2]
        padded = cv2.copyMakeBorder(image_rgb, h0 // 3, h0 // 3, w0 // 3, w0 // 3, cv2.BORDER_REPLICATE)
        results = hands.process(padded)
        if results.multi_hand_landmarks:
            image_rgb = padded

    if retry_reflect and not results.multi_hand_landmarks:
        # A smaller reflected border recovers some edge/scale failures that
        # the larger replicated border misses. Keep the normal confidence.
        h0, w0 = image_rgb.shape[:2]
        padded = cv2.copyMakeBorder(image_rgb, h0 // 6, h0 // 6, w0 // 6, w0 // 6,
                                    cv2.BORDER_REFLECT_101)
        results = hands.process(padded)
        if results.multi_hand_landmarks:
            image_rgb = padded

    n_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    return results, n_hands, image_rgb
