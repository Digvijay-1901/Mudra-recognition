"""One-command real-photo validation using the app's production predict entry point.

Photos: ROOT/ClassName/[optional session folder]/photo.jpg. Unlabelled photos
directly under ROOT are supported. --sequence replays each parent folder by file
modification time with the same 15-second smoothing as Diagnostic Mode. Use it
only for genuine consecutive captures whose original timestamps were preserved.
Without --sequence each photo is isolated, as in app uploads.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import _quiet
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
import brain
from diagnostic_temporal import smooth_capture

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FIELDS = ["File", "Expected", "Timestamp", "Raw_Label", "Raw_Confidence", "Raw_Mode",
          "Label", "Confidence", "Mode", "Hands", "Tilt", "Smoothed", "Inference_ms", "Error"]

def run(root, output, sequence=False):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Photo folder does not exist: {root}")
    output.mkdir(parents=True, exist_ok=False)
    vocabulary = {c.casefold(): c for family in brain.supported_classes().values() for c in family}
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
             and output not in p.parents]
    paths.sort(key=lambda p: (str(p.parent).casefold(), p.stat().st_mtime_ns, p.name.casefold()))
    if not paths:
        raise ValueError(f"No supported photos in {root}")
    rows = []
    state = {}
    previous_parent = None
    # Serial inference also avoids joblib's Windows worker IPC requirements.
    from gate import _model
    _model().n_jobs = 1
    with threadpool_limits(limits=1), (output/"predictions.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for i, path in enumerate(paths):
            if not sequence or previous_parent != path.parent:
                state.clear()
            previous_parent = path.parent
            relative = path.relative_to(root)
            # Accept ROOT/ClassName or ROOT itself named ClassName.
            expected = vocabulary.get(relative.parts[0].casefold()) if len(relative.parts)>1 else None
            if expected is None:
                expected = vocabulary.get(root.name.casefold(), "")
            timestamp = path.stat().st_mtime
            row = dict.fromkeys(FIELDS, "")
            row.update(File=str(relative), Expected=expected,
                       Timestamp=datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                       Raw_Label="ERROR", Raw_Mode="Unknown", Hands=0, Smoothed=False)
            started = perf_counter()
            try:
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Image cannot be decoded")
                pred = brain.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mode="auto", threshold=.30)
                row["Tilt"] = getattr(pred, "tilt_degrees", None)
                if pred is None:
                    row["Raw_Label"] = "NO_HAND"
                elif isinstance(pred, brain.Abstention):
                    row.update(Raw_Label="ABSTAIN", Hands=len(pred.results.multi_hand_landmarks or []))
                else:
                    row.update(Raw_Label=pred.label, Raw_Confidence=float(pred.confidence),
                               Raw_Mode={"Single-hand":"Single", "Double-hand":"Double"}.get(pred.mode,pred.mode),
                               Hands=pred.n_hands)
            except Exception as exc:
                row["Error"] = f"{type(exc).__name__}: {exc}"
            row["Inference_ms"] = round(1000*(perf_counter()-started),2)
            displayed = smooth_capture(state, expected or "__unlabelled__", "auto", .30, timestamp,
                                       row["Raw_Label"], row["Raw_Confidence"] if row["Raw_Confidence"] != "" else None,
                                       row["Raw_Mode"])
            row.update(Label=displayed["label"], Confidence=displayed["confidence"],
                       Mode=displayed["mode"], Smoothed=displayed["changed"])
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            if (i+1)%max(1,len(paths)//10)==0 or i+1==len(paths):
                print(f"Processed {i+1}/{len(paths)}", flush=True)
    summary = []
    for expected in sorted({r["Expected"] for r in rows if r["Expected"]}):
        subset = [r for r in rows if r["Expected"]==expected]
        summary.append(dict(Expected=expected, Images=len(subset),
                            Raw_Correct=sum(r["Raw_Label"]==expected for r in subset),
                            Displayed_Correct=sum(r["Label"]==expected for r in subset),
                            Smoothed=sum(r["Smoothed"] for r in subset)))
    with (output/"summary.csv").open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["Expected","Images","Raw_Correct","Displayed_Correct","Smoothed"])
        writer.writeheader();writer.writerows(summary)
    labelled = [r for r in rows if r["Expected"]]
    result = dict(images=len(rows),labelled=len(labelled),sequence=sequence,
                  timestamp_source="file modification time",summary=summary,
                  raw_accuracy=sum(r["Raw_Label"]==r["Expected"] for r in labelled)/len(labelled) if labelled else None,
                  displayed_accuracy=sum(r["Label"]==r["Expected"] for r in labelled)/len(labelled) if labelled else None,
                  errors=sum(bool(r["Error"]) for r in rows),
                  production_entry_point="brain.predict(mode='auto', threshold=.30)",
                  pipeline="MediaPipe gate + RTMPose single/single_norm + MediaPipe double",
                  brain_sha256=hashlib.sha256(Path(brain.__file__).read_bytes()).hexdigest(),
                  caveat="Expected labels partition sequences but never vote. Sequence replay uses original file mtimes; it is not directly comparable to independent studio-image accuracy.")
    (output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("| Expected | Photos | Raw correct | Displayed correct | Smoothed |")
    print("| --- | --- | --- | --- | --- |")
    for row in summary:
        print("| "+" | ".join(str(v) for v in row.values())+" |")
    if labelled:
        print(f"CONFIRMED raw accuracy {result['raw_accuracy']:.2%}; displayed accuracy {result['displayed_accuracy']:.2%} ({len(labelled)} labelled)")
    print(f"CONFIRMED {len(rows)} images; {result['errors']} errors; results {output}")
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos",type=Path)
    parser.add_argument("--sequence",action="store_true",help="Replay genuine capture sequences by preserved modification time")
    parser.add_argument("--output",type=Path,help="New output folder; existing folders are never overwritten")
    args=parser.parse_args()
    output=args.output or Path(__file__).resolve().parent/"reports"/("live-validation-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        run(args.photos,output.resolve(),args.sequence)
    except Exception as exc:
        parser.exit(1,f"{type(exc).__name__}: {exc}\n")

if __name__=="__main__":
    main()
