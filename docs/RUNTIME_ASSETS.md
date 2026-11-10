# Runtime assets and fresh-clone setup

CONFIRMED: production is MP routing + RTMPose single/single_norm + MP double.
No training or fine-tuning is needed to run the included trained heads.

## Install (Python 3.10)

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --progress-bar off -r requirements.txt -r tracker_v2/requirements.txt
.\venv\Scripts\python.exe -m pip check
```

The root requirements contain Streamlit/MediaPipe. The second requirements file
contains the already-used RTMLib/ONNX Runtime versions. Installing only the root
file does not install RTMPose's runtime dependencies.

## Required inference files

| Path | Delivery |
| --- | --- |
| tracker_v2/assets/rtmdet_hand.onnx | Tracked, 4,010,667 bytes |
| tracker_v2/assets/rtmpose_hand5.onnx | Download below, 55,080,248 bytes; intentionally never tracked (>50 MB) |
| tracker_v2/model_single/raw/bundle.joblib | Tracked trained single head, 330,594 bytes |
| tracker_v2/model_single/bundle.joblib | Tracked trained single_norm head, 330,226 bytes |
| gate/gate.joblib | Tracked active gate |
| model_double_v3_norm/*.joblib | Tracked MP double classifier/scaler/encoder |
| model_v3/*.joblib, model_v3_norm/*.joblib | Tracked MP-only comparison heads |
| model_trials_20261107/capped_thresholds/cutoffs.json | Tracked active policy (tracked exception inside ignored experiment directory) |

The live app uses mp.solutions.hands, with graph/model resources shipped in the
pinned MediaPipe wheel. It does NOT load an external MediaPipe .task file.
landmarks.joblib is excluded from this minimal release; the live app does not load it.

## Download the large RTMPose model

Official upstream archive:
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip

Save the archive's sole .onnx member as tracker_v2/assets/rtmpose_hand5.onnx.
Expected SHA-256: `39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29`.
The existing adapter uses the local file and performs no automatic download.

One command (run at repository root):

```powershell
.\venv\Scripts\python.exe -c "import urllib.request,zipfile,io,pathlib,hashlib; u='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip'; z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u).read())); n=[n for n in z.namelist() if n.endswith('.onnx')]; assert len(n)==1; b=z.read(n[0]); assert hashlib.sha256(b).hexdigest()=='39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29'; p=pathlib.Path('tracker_v2/assets/rtmpose_hand5.onnx'); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)"
```

## Run and validate your own photos

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
.\venv\Scripts\python.exe validate_hybrid_live.py my_photos
.\venv\Scripts\python.exe test_diagnostic_temporal.py
```

Bundled examples are included; the development outside-photo set is not.
The diagnostic suite reads its historical reference directly from
`tests/fixtures/brain.before.py`. No restoration from a removed folder is needed.
Only historical webcam replay is SKIPPED when the private file
`reports/webcam_diagnostics.csv` is absent. All synthetic smoothing, app capture,
PNG/CSV logging, landmark comparison and example-inference checks still run.

## Full acceptance: recorded development-machine evidence

CONFIRMED: 90.55% accuracy / 93.24% coverage / 39 wrong-at-95%, and outside-photo
38/40, were measured on the development machine. The five records under
reports/webcam_tripathaka_diagnosis/ are evidence of those runs, not new results
from a minimal release clone.

The raw dataset is not included. Obtain the original Jisha Raj dataset from the
source linked in README.md. Full acceptance also requires the exact frozen
landmark caches. These paths are intentionally absent from a fresh release:

- `landmarks.joblib`: frozen MediaPipe records.
- `tracker_v2/landmarks_v2.joblib`: frozen RTMPose records.
- `data/dataset/` and `data/dataset_double/`: original studio source photos,
  including the fresh source-photo comparison cohort.

`test_hybrid_production.py` is unchanged. Restore all those original inputs before
running it. Release verification reports the full acceptance check as SKIP when
they are absent; it must not pretend that the full suite ran. No training or
replacement dataset is part of release setup. The split and staged references
that are shipped do not replace the omitted landmark caches or source images.

Private capture telemetry and diagnostic PNGs are also excluded. The live app
creates a new telemetry CSV as needed; frame saving remains opt-in and OFF by
default. External MediaPipe .task files are not used. The tracked gate, trained
heads and detector remain unchanged; download the large pose ONNX as above.
