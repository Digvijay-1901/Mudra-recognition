# Bharatanatyam Mudra Recognition

Recognize Bharatanatyam hand gestures from an uploaded image or a webcam snapshot.
The current **hybrid** combines MediaPipe's hand finding and routing with RTMPose's
single-hand landmarks, geometric features and trained MLP classifiers. It runs
locally in a Streamlit app; no training is needed to use the included models.

## How predictions work

- **MediaPipe** finds hands first. Its landmarks feed the single-vs-double gate;
  the gate never receives RTMPose features.
- On a **single-hand route**, RTMDet locates the hand and RTMPose estimates its
  joints. The trained `single` or tilt-normalized `single_norm` MLP head classifies
  the geometric features.
- On a **double-hand route**, the existing MediaPipe landmarks and normalized
  double-hand MLP remain in use.
- Missing hands, an ambiguous gate or a score below the class cutoff can produce
  a retake/abstain message. Confidence is an uncalibrated model score, not a
  guarantee that an answer is correct.

## Recorded results: development machine

**CONFIRMED:** the headline **90.55% accuracy / 93.24% coverage / 39 wrong-at-95%**
and historical outside-photo **38/40** were measured on the development machine.
They are recorded under [reports/webcam_tripathaka_diagnosis/](reports/webcam_tripathaka_diagnosis/),
not re-run from this minimal release clone. Re-running full acceptance requires
the original dataset, which is not included, and both original frozen landmark caches.

| Evidence status | Check | Result |
| --- | --- | --- |
| **CONFIRMED** | Hybrid Auto, clean grouped studio validation | **90.55% accuracy / 93.24% coverage / 39 wrong-at-95%** |
| **CONFIRMED** | Historical outside-photo set (not shipped) | **38/40**: Pathaka **17/17**, Mayura **12/13**, Tripathaka **9/10** |
| **CONFIRMED** | Historical temporal-smoothing replay | Corrected **1 Pathaka** and **1 Ardhachandra** error; harmed **0** correct predictions |

**CONFIRMED evaluation scope:** the hybrid check runs the production API over
**4,465** frozen validation detector records; fresh source-image checks match on
**20/20** samples. Accuracy includes misses and abstentions as incorrect; coverage
is the proportion receiving an accepted label. Wrong-at-95% counts incorrect
accepted answers whose model score is at least **95%**.
See [acceptance evidence](reports/webcam_tripathaka_diagnosis/acceptance/acceptance.json).

**CONFIRMED outside-set scope:** the development check used AI-generated/test images and
screenshots of single-hand gestures, not an independent new-performer benchmark.
See [outside-photo results](reports/webcam_tripathaka_diagnosis/outside_test/summary.json).

The smoothing result is a replay of historical raw predictions, not a fresh live
accuracy gain. The original camera frames for that replay were not saved, so their
counterfactual predictions under the current hybrid are **UNVERIFIED**. This is a historical development-machine replay; the private telemetry is not published.

## Setup on Windows

**CONFIRMED supported setup:** use **Python 3.10**, as documented in
[Runtime assets and fresh-clone setup](docs/RUNTIME_ASSETS.md). Run these commands
from the repository root. Install **both** requirements files; the root file alone
does not install RTMPose's runtime dependencies.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --progress-bar off -r requirements.txt -r tracker_v2/requirements.txt
.\venv\Scripts\python.exe -m pip check
```

### One-time RTMPose model download

The small detector model, trained classifier heads and gate are tracked. The large
RTMPose ONNX file is intentionally excluded; download it before starting the app.

**CONFIRMED destination:** `tracker_v2/assets/rtmpose_hand5.onnx`.

**CONFIRMED SHA-256**, matching [the runtime asset specification](docs/RUNTIME_ASSETS.md):

```text
39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29
```

The command below downloads the official upstream archive, verifies the checksum
and saves its ONNX member at the exact destination. It is copied from the runtime
asset instructions:

```powershell
.\venv\Scripts\python.exe -c "import urllib.request,zipfile,io,pathlib,hashlib; u='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip'; z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u).read())); n=[n for n in z.namelist() if n.endswith('.onnx')]; assert len(n)==1; b=z.read(n[0]); assert hashlib.sha256(b).hexdigest()=='39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29'; p=pathlib.Path('tracker_v2/assets/rtmpose_hand5.onnx'); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b)"
```

A missing model prevents startup; RTMLib may report `unknown url type: c` when it
mistakes the missing Windows file path for a URL. Download the file rather than
changing the model or training anything.

### Dataset availability

The **raw studio dataset is not included**. Obtain it from the original dataset
repository linked under [Dataset source](#dataset-source). For full studio
validation and reference images, arrange the original class folders under:

```text
data/dataset/<SingleHandClass>/
data/dataset_double/<DoubleHandClass>/
```

The app can run with its bundled examples and your own photos without that dataset;
reference images will be unavailable. Full studio validation, including fresh
source-photo parity checks, requires restoring the original dataset. Ignored local
caches and private camera logs are not automatically supplied by a clone.

## Run the app

```powershell
.\venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. Choose **Auto** for the published hybrid
routing behavior, then upload an image or select **Webcam**.

In **Diagnostic Mode**, select the **Expected mudra**, then use **Capture pose for
analysis**. Expected labels are for recording/comparison; they do not vote for a
prediction. The app retains raw predictions and may display a recent consistent
prediction through temporal smoothing. That smoothing affects Diagnostic Mode's
live flow only; it does not change the core single-image prediction API.

**Save frames for debugging is OFF by default.** With it off, the CSV row is still
logged but `Frame_Path` is empty and no raw-frame PNG is written. Opt in using the
sidebar checkbox to save diagnostic images.

- Telemetry: `reports/webcam_diagnostics.csv` records the raw label, confidence,
  mode, hands, tilt, displayed label, frame path and any abstain/retake message.
- Opted-in raw images: `reports/webcam_frames/`.
- Optional landmark comparison: `analyze_webcam_frames.py` writes overlays under
  `reports/webcam_overlays/`. Frame and overlay folders are ignored by Git.

## Validate a folder of photos

`validate_hybrid_live.py` runs on the user's own photo folder.

For your own images, use class-named folders:

```text
my_photos/
  Pathaka/
    upright.jpg
  Mayura/
    example.jpg
  Tripathaka/
    example.jpg
```

```powershell
.\venv\Scripts\python.exe validate_hybrid_live.py my_photos
```

The script uses the production hybrid and prints per-class results. It writes
`predictions.csv`, `summary.csv` and `summary.json` into a new
`reports/live-validation-<timestamp>/` folder. Results include raw/displayed labels,
confidence, mode and detected hands. Flat folders can be processed without an
accuracy calculation when no true class is supplied.

For genuine consecutive captures with preserved modification times, add
`--sequence` to replay the Diagnostic Mode smoothing policy. Independent photos
remain isolated by default; copying files may change their timestamps and make a
sequence replay misleading.

## Known limitations

- **CONFIRMED live-camera failure:** Tripathaka was **1/20** on the saved webcam
  sequence, while Pathaka was **10/10**. The MediaPipe-only path also scored
  **1/20** on Tripathaka. **INFERRED:** both trackers represented the ring finger
  as nearly straight in many captures. This is evidence of a landmark/generalization
  weakness, not proof of a preprocessing bug; visual ambiguity versus landmark
  error remains **UNVERIFIED** without ground-truth joints. See the
  [final diagnosis](reports/webcam_tripathaka_diagnosis/diagnosis.md).
- **UNVERIFIED independent real-world two-hand performance:** there has been no
  dedicated two-hand validation on real-world outside photos or live camera.
  Studio validation and bundled example checks do not establish that performance.
- **CONFIRMED experiment scope:** camera captures were diagnostic only; no
  real-camera training dataset was collected or used in these experiments.
- **UNVERIFIED generalization:** studio grouping is not a held-out-person test.
  The included generated images/screenshots do not establish accuracy on new
  people, devices, lighting, backgrounds or natural hand rotations.
- Temporal smoothing can help an isolated low/moderate-confidence disagreement
  after consistent captures. It cannot recover an unseen fold or reliably correct
  a persistent confidently wrong sequence.

## Rejected experiments

These experiments were rejected; none is enabled in the production hybrid.

- **CONFIRMED — expanded anatomical features:** **80.59% Auto / 319 wrong-at-95%**, versus matched control **80.78% / 316**, on the historical pooled rotated-landmark protocol; this denominator differs from the current hybrid benchmark. The detailed historical record remains in the development backup.
- **CONFIRMED — RTMPose double-hand fallback:** **89.83% Auto / 74 wrong-at-95%**; rejected for regression. [Archived evaluation](docs/experiments/double_ensemble/evaluation.json).
- **CONFIRMED — detector fusion:** **75% single-hand detection (45/60)**, versus the hybrid's **90% (54/60)**; detection coverage, not classification accuracy. [Archived smoke comparison](docs/experiments/fused/smoke_comparison.json).

## Release test availability

The included historical fixture is `tests/fixtures/brain.before.py`.

```powershell
.\venv\Scripts\python.exe test_diagnostic_temporal.py
```

That suite still runs synthetic smoothing, capture/PNG/CSV behavior, landmark
comparison and real example inference. Its historical webcam replay prints
**SKIP** if `reports/webcam_diagnostics.csv` is absent; private telemetry is not
published. This does not disable the other checks.

`test_hybrid_production.py` is retained unchanged for full development acceptance.
In a minimal clone, **skip full acceptance** until these original inputs are
restored: `landmarks.joblib`, `tracker_v2/landmarks_v2.joblib`, `data/dataset/` and
`data/dataset_double/`. The full suite must not be reported as passed when its
inputs are absent. No substitute dataset or model is trained by this release.

## Documentation

- [Runtime assets, download, dependencies and fresh-clone requirements](docs/RUNTIME_ASSETS.md)
- [Repository cleanup inventory and dependency/grep evidence](docs/REPOSITORY_CLEANUP.md)
- [Archived experiment findings](docs/experiments/)

## Dataset source

Studio data and bundled examples come from the **Bharatanatyam Mudra Dataset by
Jisha Raj R**, collected during PhD research under Dr. Sunil T.T at the College of
Engineering, Attingal, Kerala, India. Dataset provenance for the included
outside-photo set differs from these studio examples.

Source: [Jisha Raj R's original dataset repository](https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset).
Dataset/examples licence: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
