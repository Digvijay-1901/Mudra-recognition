# Hand Landmark Gesture Recognition — Bharatanatyam Mudras

A landmark-based machine learning system for recognizing hand gestures from images, built and validated using Bharatanatyam mudras (classical Indian dance hand gestures) as the demonstration domain — 35 classes across single-hand and double-hand poses, built without a CNN or raw pixel classification.

## Overview

The core system is domain-agnostic: MediaPipe hand landmarks, hand-crafted geometric features, and a lightweight MLP classifier. Bharatanatyam mudras were chosen as the application because they offer real diversity in difficulty — from simple single-hand shapes to heavily occluded double-hand poses — making them a strong stress test for the approach.

- **25 single-hand mudras** — 96.94% held-out accuracy, 0.9537 macro F1
- **10 double-hand mudras** — 98.01% held-out accuracy, 0.9796 macro F1
- Scaled from 10 → 20 → 35 total classes over the project, with no change to the underlying architecture required at any stage — evidence the feature representation itself was sound, not tuned to a small fixed class set
- MediaPipe hand landmarks + engineered geometric features + MLP classifier — no CNN, no raw pixels
- Interactive Streamlit app with live landmark visualization, confidence breakdowns, and example images

## Architecture

```
Image (upload/webcam)
        ↓
MediaPipe Hands → 21 landmarks per hand
        ↓
Feature engineering → wrist-normalized angles, distances, inter-hand geometry
        ↓
MLP classifier (128 → 64 hidden layers) → predicted class + confidence
```

**Single-hand features (77-dim):** wrist-centered, scale-normalized landmark coordinates, fingertip-to-wrist distances, fingertip-to-fingertip distances, five joint angles.

**Double-hand features (162-dim):** the same per-hand shape features for each hand, plus inter-hand relational features (wrist distance, palm distance, fingertip-to-fingertip distances) computed in a shared coordinate frame, plus an explicit flag indicating whether both hands were actually detected or one was reconstructed from partial data.

### Why two separate pipelines (single-hand vs. double-hand)

Double-hand mudras need relational math between the two hands — distances and angles that simply don't exist for a single-hand pose. Forcing both cases through one shared feature vector would mean padding single-hand images with meaningless placeholder values for features they don't have, and would tie the two problems together when they benefit from independent tuning. Keeping them as separate feature extractors, training scripts, and models let the double-hand side get more sophisticated (relational features, occlusion-completeness flags) without diluting or complicating the simpler single-hand path.

## Routing logic — important note for anyone extending this

Single-hand and double-hand images are routed to different models, but **the routing decision is not a simple hand-count check** — an earlier, simpler version of this logic caused two separate real bugs during development (see Challenges), so the exact decision order matters:

1. MediaPipe detects exactly 2 hands → **double-hand model** (trusted directly)
2. Otherwise → **try the single-hand model first**
   - If single-hand confidence ≥ 0.50 → use that result
   - If single-hand confidence < 0.50 → try the double-hand model instead
     (this catches double-hand mudras where MediaPipe only found 1 hand due to occlusion — e.g. Shanka, Kurma)
3. If nothing above produced a usable result → fall back to whatever single-hand result is available, even at low confidence

**Why this specific order matters:** the double-hand feature extractor can produce a technically valid feature vector using just *one* detected hand (the missing hand is zero-padded, with an explicit `both_detected_flag = 0`). This means "did feature extraction succeed" is *not* the same thing as "were two hands actually detected" — a naive router that only checks for a successful double-hand feature vector will misroute a large fraction of genuine single-hand images into the double-hand model, because the double-hand extractor doesn't reject single-hand input on its own.

Two specific failure modes were hit and fixed during development:
- **Routing based on raw MediaPipe hand-count alone** sent genuine double-hand images to the single-hand model when MediaPipe detected only 1 hand on occluded double-hand poses.
- **Routing based on "did double-hand extraction succeed" alone** (trying double-hand first, always) caused genuine single-hand mudras to be misclassified by the double-hand model, since a single detected hand alone is enough for the double-hand extractor to produce a "valid" (but essentially half-empty) result.

The current confidence-based, single-hand-first ordering resolves both failure modes. **Any diagnostic or test script evaluating routing accuracy must reproduce this exact logic** — a simplified stand-in (e.g. checking `both_detected_flag` alone) will produce misleading numbers that look like a regression but aren't, since it's testing a different rule than what's actually in production.

## Why not a CNN?

A CNN approach was tried directly on cropped hand images early on. Two problems ruled it out: many mudras differ only in subtle finger curl or spacing, and a CNN's pixel-level features are a black box — when it confused visually close classes, there was no way to inspect *why*, or which part of the hand it was even attending to. Switching to explicit landmark-based geometric features solved both problems at once: every feature has a known physical meaning (a specific angle, a specific distance), so when two classes get confused, it's possible to look at the actual numbers and see exactly which measurement isn't separating them — rather than guessing at a black box.

## Does this generalize beyond mudras?

The methodology was validated directly, not just claimed: the class count grew from an initial 10 to 20 to 35 over the project — including several double-hand poses with genuinely difficult occlusion properties — and the pipeline trained every new batch successfully without any architecture changes, maintaining 96–98% held-out accuracy throughout.

The current trained model recognizes exactly 35 specific classes and nothing else — like any multiclass classifier, it cannot detect or flag an unfamiliar gesture as "unknown"; it always outputs its closest guess among the 35 it knows. Extending it to new gesture types is a matter of adding labeled images and retraining, not redesigning the system.

## Results

| Model | Classes | Held-out accuracy | Macro F1 |
|---|---|---|---|
| Single-hand | 25 | 96.94% | 0.9537 |
| Double-hand | 10 | 98.01% | 0.9796 |

Evaluated on a stratified 80/20 train/test split — the model is never trained on the images used to compute these numbers.

**Known confusable pairs:** Shukatundam/Aralam (single-hand's weakest class, 0.78 F1) and Ardhapathaka/Tripathaka show the most classification overlap; Anjali/Kapotham and a few pairs among the newer double-hand classes (Kurma/Pushpaputa, Shanka/Shivalinga) show smaller, expected confusion.

## Challenges and what was actually debugged

This project's engineering value is as much in the debugging as the pipeline itself:

**Limited training data.** No existing labeled dataset covers Bharatanatyam mudras at this scale — every image had to be sourced and labeled from scratch, for all 35 classes. This capped how much data could go toward underrepresented or harder classes and is the main reason held-out accuracy is reported carefully rather than treated as proof of full real-world generalization (see Limitations).

**Raw MediaPipe landmarks weren't enough on their own.** The 21 raw hand keypoints, used directly, produced weak and inconsistent classification — they don't inherently encode which spatial relationships actually distinguish one mudra from another. Getting a usable signal took many rounds of adding one engineered feature at a time (specific distances, specific angles, curl measurements) and testing which ones actually separated confusable mudras versus which just added noise. The real breakthrough was structuring the math so it reliably outputs one consistent, scale-and-position-normalized numeric signature per hand pose — once that structure was solid, adding new mudra classes (10 → 20 → 35) stopped requiring any rework of the feature pipeline itself.

**Silent scaler mismatch.** An earlier version fit a `StandardScaler` during training but trained the classifier on raw (unscaled) features, while inference applied the scaler — a train/inference contract violation that produced ~97% offline accuracy alongside near-random live predictions. Traced with a targeted raw-vs-scaled prediction test on a known training example.

**Stale model vs. stale evidence.** Discovered a deployed model file dated ten days *after* the confusion matrix being used to trust it — the trusted evaluation belonged to a different training run entirely.

**Evaluation methodology bug.** An earlier evaluation script tested the model against the full dataset it trained on, reporting an inflated, meaningless "accuracy." Rebuilt with a proper held-out split.

**MediaPipe detection failures on merged/occluded hands.** Several double-hand poses (crossed, stacked, or pressed-together hands) frequently return fewer than 2 hand detections from MediaPipe's palm detector — a structural detector limitation, not a data quality issue. Solved with graceful degradation (zero-padding the missing hand's features plus an explicit detection-completeness flag) rather than discarding those images.

**Live-app routing.** Deciding which model (single-hand or double-hand) should handle a given image turned out to be its own problem — a naive hand-count check isn't reliable, since MediaPipe sometimes only detects one hand in a genuine double-hand pose, and can occasionally over-detect in a single-hand image. The routing logic settled on a two-stage strategy: trust a clear 2-hand detection outright, otherwise attempt single-hand classification first and only defer to the double-hand model when that single-hand prediction is itself unconfident.

## Live webcam vs. upload

Continuous live-webcam inference was explored and works in principle — the underlying MediaPipe + feature pipeline doesn't care whether a frame comes from an upload or a camera stream. In practice, reliable low-latency continuous webcam capture inside Streamlit depends on extra components (e.g. `streamlit-webrtc`) with real browser/deployment quirks that go beyond this project's scope. The shipped app is built around image upload and single-snapshot capture instead, which is reliable across environments; wiring up true continuous live inference is additional integration work, not a change to the recognition system itself.

## Limitations

- **Evaluation is sample-level, not person-level.** The held-out split separates images, not performers/sessions — the reported accuracy reflects generalization to unseen photos from the same dataset, not confirmed generalization to a new person's hand under new conditions. Dataset composition (distinct performers, sessions, or devices represented) has not been formally characterized. Informal testing on out-of-dataset photos (different background, lighting, hand) suggests reasonable real-world generalization, but this hasn't been rigorously validated.
- **Confidence scores are model probability estimates**, not calibrated real-world certainty.
- **No out-of-distribution rejection.** The classifier always predicts one of its known classes; it has no mechanism to flag a genuinely unfamiliar gesture as "unknown."
- A small number of confusable class pairs (listed above) remain the main source of classification error in each model.

## Project structure

```
build_keypoints_csv.py           # builds single-hand training CSV from raw images
build_keypoints_csv_double.py    # builds double-hand training CSV from raw images
feature_extraction.py            # single-hand landmark → feature vector
feature_extraction_double.py     # double-hand landmark → feature vector (incl. relational features)
train_mlp.py                     # trains single-hand MLP classifier
train_mlp_double.py              # trains double-hand MLP classifier
evaluate_model.py                # held-out accuracy/F1 for single-hand model
evaluate_model_double.py         # held-out accuracy/F1 for double-hand model
confusion_graph.py               # confusion matrix visualization, single-hand
confusion_graph_double.py        # confusion matrix visualization, double-hand
predict_image.py                 # CLI single-hand prediction
predict_image_double.py          # CLI double-hand prediction
streamlit_app.py                 # interactive web app
model/ , model_double/           # trained classifier, scaler, label encoder artifacts
data/                            # training images (not included in repo)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows / source venv/bin/activate on macOS-Linux
pip install -r requirements.txt
```

## Usage

```bash
# Build the training data from images in data/dataset and data/dataset_double
python build_keypoints_csv.py
python build_keypoints_csv_double.py

# Train
python train_mlp.py
python train_mlp_double.py

# Evaluate
python evaluate_model.py
python evaluate_model_double.py
python confusion_graph.py
python confusion_graph_double.py

# Predict a single image from the command line
python predict_image.py path/to/image.jpg
python predict_image_double.py path/to/image.jpg

# Run the interactive app
streamlit run streamlit_app.py
```

## Tech stack

MediaPipe · scikit-learn · Streamlit · OpenCV · NumPy · pandas

## Acknowledgements

Built on top of Google's MediaPipe Hands for landmark detection.
