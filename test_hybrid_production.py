"""Production-entry acceptance: frozen clean split replay + fresh pixel checks.

Only detector outputs are replayed for the full 4,465-image cohort. Actual
brain.predict routing, geometry, models and cutoffs run for every record. Fresh
inference checks use the same 20 source photos as the original staged test.
No training, policy search, or changes to staged evidence.
"""
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import _quiet
import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import brain
import hybrid_single
from gate import _model

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "tracker_v2/hybrid_candidate"
OUT = ROOT / "reports/hybrid_live_wiring"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(record):
    return np.broadcast_to(np.uint8(0), (record["h"], record["w"], 3))


def landmarks(record):
    return NS(multi_hand_landmarks=[NS(landmark=[NS(x=float(x), y=float(y), z=float(z))
                                               for x, y, z in hand]) for hand in record["hands"]],
              multi_handedness=[NS(classification=[NS(label=label)]) for label in record["handed"]])


def label(prediction):
    if prediction is None:
        return "NO_HAND"
    return "ABSTAIN" if isinstance(prediction, brain.Abstention) else prediction.label


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((STAGE / "manifest.json").read_text())
    protected = {name: digest for name, digest in manifest["runtime_inputs"].items() if name != "brain.py"}
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest, "Staged runtime input changed: " + name
    meta = json.loads((ROOT / "tracker_v2/features_manifest.json").read_text())
    assert meta["complete"] and meta["fingerprint"] == manifest["split_fingerprint"]
    assert sha(ROOT / "landmarks.joblib") == meta["source_sha256"]
    for name, digest in meta["input_hashes"].items():
        assert sha(ROOT / "tracker_v2" / name) == digest
    old = joblib.load(ROOT / "landmarks.joblib")
    new = joblib.load(ROOT / "tracker_v2/landmarks_v2.joblib")
    split = joblib.load(ROOT / "tracker_v2/split_v2.joblib")
    assert split["fingerprint"] == manifest["split_fingerprint"]
    with (STAGE / "validation_predictions.csv").open(newline="", encoding="utf8") as handle:
        expected = list(csv.DictReader(handle))
    assert [int(row["Source_Index"]) for row in expected] == list(split["validation"])
    _model().n_jobs = 1
    observed = []
    single_calls = gate_calls = 0
    for position, row in enumerate(expected):
        index = int(row["Source_Index"])
        mp, rt = old[index], new[index]
        assert (mp["group"], mp["cls"], mp["file"]) == (rt["group"], rt["cls"], rt["file"])
        image, rt_frame = frame(mp), frame(rt)
        mp_result = landmarks(mp)
        valid_gate = []
        if mp["n_hands"]:
            hand = mp_result.multi_hand_landmarks[0]
            valid_gate = [brain.single_hand_features_from_landmarks(hand, image.shape),
                          brain.normalized_single_hand_features(hand, image.shape)]
        original_gate = brain.is_double

        def checked_gate(features):
            nonlocal gate_calls
            assert any(np.array_equal(features, allowed) for allowed in valid_gate), "Non-MediaPipe gate input"
            gate_calls += 1
            return original_gate(features)

        fake = NS(detect=lambda bgr: (rt["hands"], [], rt_frame))
        with patch("brain.detect", return_value=(mp_result, mp["n_hands"], image)), \
                patch("brain.is_double", side_effect=checked_gate), \
                patch("hybrid_single.tracker", return_value=fake) as tracker_call:
            prediction = brain.predict(image)
            single_calls += tracker_call.call_count
            assert bool(tracker_call.call_count) == (row["Route"] == "single")
        actual = label(prediction)
        assert actual == row["Predicted"], (index, actual, row["Predicted"])
        accepted = isinstance(prediction, brain.Prediction)
        assert accepted == (row["Accepted"].lower() == "true")
        if accepted:
            assert np.isclose(prediction.confidence, float(row["Confidence"]), atol=1e-5, rtol=0)
        observed.append(dict(source_index=index, true=mp["cls"], label=actual, accepted=accepted,
                             confidence=prediction.confidence if accepted else 0.))
        if (position + 1) % 1000 == 0:
            print("Production entry cached-landmark replay", position + 1, flush=True)
    correct = sum(r["accepted"] and r["label"] == r["true"] for r in observed)
    metrics = dict(n=len(observed), correct=correct, accuracy=correct / len(observed),
                   coverage=sum(r["accepted"] for r in observed) / len(observed),
                   wrong95=sum(r["accepted"] and r["label"] != r["true"] and r["confidence"] >= .95 for r in observed))
    previous = json.loads((STAGE / "results.json").read_text())
    assert metrics == previous["hybrid"], (metrics, previous["hybrid"])
    print("CONFIRMED production entry full clean split:", metrics, flush=True)
    # Unmocked detectors and real source pixels, same cohort as original runtime check.
    fresh = []
    for row in previous["runtime_checks"]:
        index = row["source_index"]
        record = old[index]
        tree = "dataset" if record["group"] == "single" else "dataset_double"
        path = ROOT / "data" / tree / record["cls"] / record["file"]
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert bgr is not None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with patch("hybrid_single.classify", wraps=hybrid_single.classify) as classify:
            prediction = brain.predict(rgb, threshold=.30)
        actual = label(prediction)
        assert actual == row["cached_label"], (index, actual, row["cached_label"])
        assert bool(classify.call_count) == (row["cached_route"] == "single")
        fresh.append(dict(source_index=index, label=actual, route=row["cached_route"],
                          rtmpose_called=bool(classify.call_count)))
    # This guards the validator's actual imported entry point as well as future drift.
    import validate_hybrid_live
    assert validate_hybrid_live.brain.predict is brain.predict
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest
    report = dict(status="CONFIRMED PASS", method="production brain.predict; cached detector outputs on all validation records; fresh pixels separately",
                  metrics=metrics, split_fingerprint=manifest["split_fingerprint"], gate_input="MediaPipe only",
                  gate_calls=gate_calls, rtmpose_single_calls=single_calls, fresh_checks=fresh,
                  protected_inputs_unchanged=len(protected), validator_entrypoint="brain.predict",
                  runtime_sources={name: sha(ROOT / name) for name in ("brain.py", "hybrid_single.py", "streamlit_app.py", "validate_hybrid_live.py")})
    (OUT / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print("PASS 4465/4465 staged decisions; MediaPipe-only gate; fresh pixels 20/20; validator shares production API", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
