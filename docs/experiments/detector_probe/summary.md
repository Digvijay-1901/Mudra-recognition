# Investigation only

## 1. Gate correctness
CONFIRMED by source inspection and controlled call interception: raw MediaPipe single-hand features enter gate.is_double first. The existing >=75 degree ambiguous-gate retry receives normalized MediaPipe features. RTMPose detection occurs only after the single route is selected. No RTMPose features enter the gate. The normalized retry is also present in production brain.py and the cached evaluator, so this check does not revise the previously reported 90.55%. Its new-person reliability remains UNVERIFIED. No gate retraining is needed to fix a tracker-format mismatch.

## 2. Fixed training-only detector probe
100 double photos (10 per class) and 50 single controls selected once with seed 20261108 from the established training split. Source record cache and photos were necessary for fresh detection. No validation/test or capture photos selected.
CONFIRMED both accepted hand outputs on double photos: default .30 detector threshold 18/100; .15 threshold 33/100; .15 plus retry below two hands 51/100; matched MediaPipe 23/100.
CONFIRMED two outputs on single controls: default 0/50; lower threshold 1/50; padded retry 23/50; MediaPipe 0/50.
Landmark quality thresholds remain .30. Existing default retries only on zero hands. Padded variant returns the maximum count from one pass, never merges passes.
INFERRED: lower threshold is promising for detection recall. Padded retry below two hands is unsuitable as implemented because it produces many false two-hand outputs, potentially reflected duplicates. Counts alone do not certify localization of two distinct true hands. This probe does not establish a full 6355-photo rate, Auto accuracy, or a promotable policy.
CONFIRMED mean measured latency: baseline 105.9ms versus combined lower-threshold/retry 274.0ms on this probe. These are separate from the earlier hybrid benchmark.
First execution completed inference but failed JSON serialization; corrected NumPy scalar serialization and reran once, adding matched MP controls. Second execution succeeded.

## 3. Mixed detector / landmarker feasibility
CONFIRMED locally: MediaPipe Hands.process accepts an image only, with no external box input. RTMPose pose_model accepts bboxes. Current MediaPipe wrapper exposes final landmarks, not an independent palm detector API.
INFERRED: RTMDet-to-MediaPipe landmarker-only integration requires graph/ROI integration and crop coordinate/Z validation. Feeding crops to Hands.process would also rerun its detector. The reverse requires exposing MediaPipe palm detections, or first computing MP landmarks to derive boxes (which already incurs the complete MP pipeline). This exceeds a quick isolated component swap in the current wrappers.
UNVERIFIED: five-pair balanced accuracy of either mixed pipeline. No mixed pipeline was built.

No production promotion. Original candidate runtime manifest matches at completion.
