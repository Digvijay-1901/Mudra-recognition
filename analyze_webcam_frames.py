r"""Read-only inference comparison; writes only new reports/overlays.

Run: .\venv\Scripts\python.exe analyze_webcam_frames.py
MediaPipe control = pre-hybrid production router, v3 raw/normalized single
heads, current calibrated MP double head and unchanged .30 class-cutoff policy.
Never trains, changes thresholds, or writes to input frames/models/telemetry.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import _quiet
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
import brain
import hybrid_single
from detector import detect
from gate import _model

ROOT = Path(__file__).resolve().parent
FIELDS = ["Frame", "Expected", "Hybrid_Label", "Hybrid_Confidence", "Hybrid_Mode", "Hybrid_Hands",
          "Hybrid_Message", "MP_Label", "MP_Confidence", "MP_Mode", "MP_Message",
          "RTMDet_Boxes", "RTMPose_Hands", "MediaPipe_Hands", "RTMDet_Attempt_Counts", "Overlay", "Error"]
EDGES = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)] + [
    (start + offset, start + offset + 1) for start in (1, 5, 9, 13, 17) for offset in range(3)]


def mediapipe_only(image_rgb, mode="auto", threshold=.30):
    """Pre-promotion MP policy, including its ambiguous tilted-gate retry."""
    results, n, used = detect(image_rgb)
    if not n:
        return None
    hand = results.multi_hand_landmarks[0]
    tilt = brain.hand_tilt_degrees(hand, used.shape)
    normalized = tilt is not None and abs(tilt) >= brain.NORMALIZE_AT_DEGREES
    route = mode
    if route == "auto" and n == 1:
        features = brain.single_hand_features_from_landmarks(hand, used.shape)
        try:
            _, score = brain.is_double(features) if features is not None else (False, .50)
            if normalized and .4 < score < .6:
                _, score = brain.is_double(brain.normalized_single_hand_features(hand, used.shape))
        except FileNotFoundError:
            return brain.Abstention("The routing gate is missing. Restore gate/gate.joblib or choose a hand mode.", results, used, tilt)
        if .4 < score < .6:
            return brain.Abstention("Only one hand detected. Show both hands for a two-hand mudra, or choose Single-hand mode if appropriate.", results, used, tilt)
        route = "double" if score >= .6 else "single"
    elif route == "auto":
        route = "double"
    if route == "single":
        family = "single_norm" if normalized else "single"
        features = (brain.normalized_single_hand_features(hand, used.shape) if normalized
                    else brain.single_hand_features_from_landmarks(hand, used.shape))
        res = brain._classify("model_v3_norm" if normalized else "model_v3", features)
        label_mode = "Single-hand"
    else:
        family = "double"
        res = brain._classify("model_double_v3_norm", brain.normalized_double_hand_features(results, used.shape))
        label_mode = "Double-hand"
    if res and res[1] >= brain.decision_threshold(family, res[0], threshold):
        label, confidence, probabilities, names = res
        return brain.Prediction(label, confidence, label_mode, probabilities, names, n, results, used, tilt)
    return brain.Abstention("Unsure at the selected confidence threshold. Try a clearer photo.", results, used, tilt)


def prediction_fields(value):
    if value is None:
        return dict(label="NO_HAND", confidence="", mode="Unknown", hands=0,
                    message="No hand detected — try a clearer, well-lit image.")
    if isinstance(value, brain.Abstention):
        return dict(label="ABSTAIN", confidence="", mode="Unknown",
                    hands=len(value.results.multi_hand_landmarks or []), message=value.message)
    return dict(label=value.label, confidence=value.confidence,
                mode=value.mode.replace("-hand", ""), hands=value.n_hands, message="")


def rt_detection(bgr):
    """Observe real detector boxes without altering adapter logic or its models."""
    attempts = []
    with hybrid_single._lock:
        tracker = hybrid_single.tracker()
        original = tracker.model.det_model

        def observe(image):
            boxes = original(image)
            attempts.append(dict(shape=list(image.shape), boxes=np.asarray(boxes).tolist()))
            return boxes

        with patch.object(tracker.model, "det_model", side_effect=observe):
            hands, scores, used = tracker.detect(bgr)
    return hands, scores, used, attempts


def original_points(hands, used_shape, original_shape):
    # Both production adapters pad symmetrically and never resize their returned
    # image. Undo that padding before drawing on the original camera pixels.
    h, w = used_shape[:2]
    top = (h - original_shape[0]) // 2
    left = (w - original_shape[1]) // 2
    return [np.asarray(hand)[:, :2] * [w, h] - [left, top] for hand in hands]


def paint(image, hands, color, title, boxes=()):
    panel = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 1)
    for index, points in enumerate(hands, 1):
        points = np.rint(points).astype(int)
        for a, b in EDGES:
            cv2.line(panel, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
        for joint, point in enumerate(points):
            cv2.circle(panel, tuple(point), 3, color, -1)
            cv2.putText(panel, str(joint), tuple(point + [3, -3]), cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
        cv2.putText(panel, f"hand {index}", tuple(points[0] + [4, 15]), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(panel, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return panel


def run(frames, output, telemetry=None):
    frames, output = Path(frames).resolve(), Path(output).resolve()
    if output == frames or frames in output.parents or output in frames.parents:
        raise ValueError("Output and input frame directories must be separate")
    paths = sorted(frames.rglob("*.png")) if frames.is_dir() else []
    if not paths:
        print(f"CONFIRMED no saved webcam PNG frames in {frames}. Capture with Webcam > Diagnostic Mode > Capture pose for analysis.")
        return []
    if len(paths) > 200:
        print(f"Processing {len(paths)} saved frames; inference only, no training.", flush=True)
    metadata = {}
    if telemetry is not None:
        telemetry = Path(telemetry)
        for path in [*sorted(telemetry.parent.glob(telemetry.stem + ".archive-*.csv")), telemetry]:
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    if row.get("Frame_Path"):
                        metadata[str(Path(row["Frame_Path"]).resolve()).casefold()] = row
    output.mkdir(parents=True, exist_ok=False)
    _model().n_jobs = 1  # In-memory only; avoid Windows joblib worker creation.
    rows, details = [], []
    with threadpool_limits(limits=1), (output / "comparison.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, path in enumerate(paths):
            row = dict.fromkeys(FIELDS, "")
            row.update(Frame=str(path), Expected=metadata.get(str(path).casefold(), {}).get("Expected_Class", ""))
            try:
                bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Cannot decode saved PNG")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                hybrid = prediction_fields(brain.predict(rgb, threshold=.30))
                mp = prediction_fields(mediapipe_only(rgb))
                mp_result, mp_n, mp_used = detect(rgb)
                hands, scores, rt_used, attempts = rt_detection(bgr)
                mp_hands = [[[p.x, p.y, p.z] for p in hand.landmark] for hand in (mp_result.multi_hand_landmarks or [])]
                rp = original_points(hands, rt_used.shape, bgr.shape)
                mp_points = original_points(mp_hands, mp_used.shape, bgr.shape)
                boxes = []
                if attempts:
                    shape = attempts[-1]["shape"]
                    shift = np.array([(shape[1] - bgr.shape[1]) // 2, (shape[0] - bgr.shape[0]) // 2] * 2)
                    boxes = [np.asarray(box[:4]) - shift for box in attempts[-1]["boxes"]]
                overlay = np.hstack((paint(bgr, rp, (0, 255, 0), f"RTMPose: {len(hands)} hands", boxes),
                                     paint(bgr, mp_points, (255, 0, 255), f"MediaPipe: {mp_n} hands")))
                target = output / f"{index + 1:05d}-{path.stem}.png"
                ok, encoded = cv2.imencode(".png", overlay)
                if not ok:
                    raise OSError("Overlay PNG encoding failed")
                with target.open("xb") as overlay_file:
                    overlay_file.write(encoded.tobytes())
                row.update(Hybrid_Label=hybrid["label"], Hybrid_Confidence=hybrid["confidence"], Hybrid_Mode=hybrid["mode"],
                           Hybrid_Hands=hybrid["hands"], Hybrid_Message=hybrid["message"], MP_Label=mp["label"],
                           MP_Confidence=mp["confidence"], MP_Mode=mp["mode"], MP_Message=mp["message"],
                           RTMDet_Boxes=len(boxes), RTMPose_Hands=len(hands), MediaPipe_Hands=mp_n,
                           RTMDet_Attempt_Counts="/".join(str(len(a["boxes"])) for a in attempts), Overlay=str(target))
                details.append(dict(frame=str(path), rtmpose_xy_original=[p.tolist() for p in rp],
                                    rtmpose_joint_scores=[np.asarray(s).tolist() for s in scores],
                                    mediapipe_xy_original=[p.tolist() for p in mp_points], detector_attempts=attempts,
                                    capture_telemetry=metadata.get(str(path).casefold(), {})))
            except Exception as error:
                row["Error"] = f"{type(error).__name__}: {error}"
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if (index + 1) % max(1, len(paths) // 10) == 0:
                print(f"Processed {index + 1}/{len(paths)}", flush=True)
    report = dict(status="CONFIRMED inference comparison; no causal attribution from scores alone", frames=len(rows),
                  errors=sum(bool(row["Error"]) for row in rows), policy="Auto, floor .30, no temporal replay; per-frame raw inference",
                  count_source="INFERRED: brain.py single route assigns n_hands from hybrid_single.classify=len(Tracker.detect hands); double route keeps detector.detect's MediaPipe landmark-list length. RTMDet_Boxes is final attempted detector box count before keypoint filtering; RTMPose_Hands is retained count after filtering/top-2 selection.",
                  why_two="INFERRED: RTMDet may supply duplicate/spurious boxes; adapter.py retains up to two qualifying poses with no one-physical-hand constraint. Single classification uses hands[0] while reporting len(hands). MediaPipe can also return spurious/duplicate landmark sets. Which happened in the earlier Tripathaka frames is UNVERIFIED because those pixels were not saved.",
                  details=details)
    (output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print("| Frame | Hybrid label | Score | MP label | Score | RTMDet boxes | RTMPose hands | MP hands |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows[-40:]:
        def score(value):
            return f"{value:.1%}" if isinstance(value, (int, float)) else "--"
        print(f"| {Path(row['Frame']).name} | {row['Hybrid_Label'] or 'ERROR'} | {score(row['Hybrid_Confidence'])} | {row['MP_Label']} | {score(row['MP_Confidence'])} | {row['RTMDet_Boxes']} | {row['RTMPose_Hands']} | {row['MediaPipe_Hands']} |")
    if len(rows) > 40:
        print("Last 40 rows shown; comparison.csv contains every frame.")
    print(f"CONFIRMED {len(rows)} frames; {report['errors']} errors; overlays, comparison.csv and analysis.json: {output}")
    print(report["count_source"])
    print(report["why_two"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=ROOT / "reports/webcam_frames")
    parser.add_argument("--telemetry", type=Path, default=ROOT / "reports/webcam_diagnostics.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "reports/webcam_overlays" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        rows = run(args.frames, output, args.telemetry)
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    if any(row["Error"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
