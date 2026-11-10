"""Diagnostic CSV schema, serialized writes and opt-in raw-frame persistence."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import streamlit as st


BASE = Path(__file__).resolve().parent

DIAGNOSTICS_CSV = BASE / "reports" / "webcam_diagnostics.csv"

DIAGNOSTIC_FRAMES = BASE / "reports" / "webcam_frames"

DIAGNOSTIC_FIELDS = [
    "Timestamp", "Expected_Class", "Predicted_Class",
    "Confidence", "Mode_Detected", "Hands_Count", "Tilt_Degrees",
    "Frame_Path", "Displayed_Label", "Message",
]

@st.cache_resource(show_spinner=False)
def load_diagnostic_lock():
    return Lock()

def append_diagnostic_row(row=None):
    """One process lock covers schema rollover, header and complete rows."""
    with load_diagnostic_lock():
        if row is not None and len(row) != len(DIAGNOSTIC_FIELDS):
            raise ValueError("Diagnostic row does not match the current schema")
        DIAGNOSTICS_CSV.parent.mkdir(parents=True, exist_ok=True)
        if DIAGNOSTICS_CSV.exists() and DIAGNOSTICS_CSV.stat().st_size:
            with DIAGNOSTICS_CSV.open(newline="", encoding="utf-8-sig") as previous:
                reader = csv.reader(previous)
                header = next(reader, None)
                has_rows = header != DIAGNOSTIC_FIELDS and any(any(cell.strip() for cell in item) for item in reader)
            if header != DIAGNOSTIC_FIELDS:
                if header == DIAGNOSTIC_FIELDS[:-1] and not has_rows:
                    # Preserve the existing file when it contains only the legacy header.
                    with DIAGNOSTICS_CSV.open("w", newline="", encoding="utf-8") as output:
                        csv.writer(output).writerow(DIAGNOSTIC_FIELDS)
                else:
                    # Reader is closed before renaming, including on Windows.
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    archive = DIAGNOSTICS_CSV.with_name(f"{DIAGNOSTICS_CSV.stem}.archive-{stamp}.csv")
                    sequence = 0
                    while archive.exists():
                        sequence += 1
                        archive = DIAGNOSTICS_CSV.with_name(f"{DIAGNOSTICS_CSV.stem}.archive-{stamp}-{sequence}.csv")
                    DIAGNOSTICS_CSV.rename(archive)
        with DIAGNOSTICS_CSV.open("a", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            if output.tell() == 0:
                writer.writerow(DIAGNOSTIC_FIELDS)
            if row is not None:
                writer.writerow(row)

def save_diagnostic_frame(image, timestamp):
    """Write unannotated pixels once, using exclusive names to protect prior captures."""
    import cv2
    (ok, png) = cv2.imencode('.png', image)
    if not ok:
        raise OSError('PNG encoding failed')
    stamp = datetime.fromisoformat(timestamp).strftime('%Y%m%dT%H%M%S%f%z')
    with load_diagnostic_lock():
        DIAGNOSTIC_FRAMES.mkdir(parents=True, exist_ok=True)
        suffix = 0
        while True:
            path = DIAGNOSTIC_FRAMES / (stamp + (f'-{suffix}' if suffix else '') + '.png')
            try:
                with path.open('xb') as output:
                    output.write(png.tobytes())
                return str(path.resolve())
            except FileExistsError:
                suffix += 1
