"""UI-only raw-capture history. Expected class is a partition key, never a vote."""
from collections import Counter
import math

# Vote from recent raw captures only; expected labels partition history and never contribute votes.
def smooth_capture(state, expected, mode, threshold, timestamp, label, confidence, detected_mode):
    context = (expected, mode, float(threshold))
    if state.get("context") != context:
        state.clear()
        state["context"] = context
    history = [r for r in state.get("history", [])
               if 0 <= timestamp-r["time"] <= 15][-4:]
    result = dict(label=label, mode=detected_mode, confidence=confidence, changed=False)
    valid = confidence is not None and math.isfinite(confidence) and detected_mode in ("Single", "Double")
    if valid and confidence <= .85 and len(history) >= 2:
        votes = Counter((r["label"], r["mode"]) for r in history
                        if r["confidence"] is not None and r["confidence"] >= .60)
        if votes:
            winner, count = votes.most_common(1)[0]
            recent = [(r["label"], r["mode"]) for r in history[-2:]]
            if count >= 2 and count > len(history)/2 and recent == [winner, winner] and winner != (label, detected_mode):
                supporters = [r["confidence"] for r in history
                              if (r["label"], r["mode"]) == winner and r["confidence"] is not None]
                result.update(label=winner[0], mode=winner[1], confidence=min(supporters), changed=True)
    history.append(dict(time=float(timestamp), label=label, mode=detected_mode,
                        confidence=confidence if valid else None))
    state["history"] = history[-5:]
    return result
