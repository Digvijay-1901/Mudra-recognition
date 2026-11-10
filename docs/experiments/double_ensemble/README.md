# Finished double-hand fallback candidate

CONFIRMED admission FAIL. This closes the requested double-hand attempt. Candidate is integrated behind Hybrid(double_ensemble=True); Hybrid() preserves the original staged hybrid. No promotion.

## Training-only grid

400 fixed training photos: 200 double (20/class), 200 single controls. Thresholds use identical images without padding. Selection maximizes mean two-hand recall and single-hand specificity; ties prefer fewer false twos then higher threshold. Padding is tested only after selection and retained only with an improved score and no false-two increase. Counts are accepted landmark-set counts, not annotated distinct-hand localization.

| Threshold | Padding | Double two outputs | Single false two | Balanced score | Mean detector+landmarker ms |
| --- | --- | --- | --- | --- | --- |
| 0.15 | False | 38.5% | 6.5% | 66.00% | 184.0 |
| 0.20 | False | 28.5% | 3.0% | 62.75% | 104.7 |
| 0.25 | False | 23.0% | 1.5% | 60.75% | 77.0 |
| 0.30 | False | 15.5% | 0.5% | 57.50% | 67.6 |
| 0.15 | True | 59.0% | 45.0% | 57.00% | 322.1 |

Frozen detector threshold .15, padding disabled. Confident proposal requires two existing-quality landmark sets and >=.95 double-class probability. Existing trained RTMPose double head and normalized geometry are used. Any rejected proposal falls back to the existing MP double result. The single path and gate policy are inherited from the original hybrid.

## Full clean validation

4,465 matched sources, same established split. Baseline CSV decisions are reused for unchanged paths; fresh tuned RTMPose inference is run on all 1,304 double routes. This follows the cached-feature plus fresh-check methodology of the original hybrid result. No validation tuning or retraining. Forty matched end-to-end live calls (20 single routes and 20 double) agree with baseline/candidate caches.

| Arm | Accuracy | Coverage | Wrong-at-95% |
| --- | --- | --- | --- |
| Existing hybrid | 90.55% | 93.24% | 39 |
| Finished double fallback | 89.83% | 93.24% | 74 |

399 confident RTMPose proposals; 5 rescued baseline errors/abstentions and 37 spoiled baseline correct answers. 37 newly confident wrong outcomes, 2 previous confident wrong outcomes removed.

| Protected class | Baseline accuracy | Candidate accuracy |
| --- | --- | --- |
| Anjali | 96.46% | 95.58% |
| Chakra | 97.56% | 96.34% |
| Garuda | 97.67% | 96.12% |
| Kapotham | 98.58% | 96.45% |
| Karkatta | 96.32% | 93.87% |
| Nagabandha | 100.00% | 97.52% |
| Pushpaputa | 96.15% | 90.38% |
| Shanka | 98.17% | 96.95% |
| Shivalinga | 97.80% | 96.70% |

## Latency and admission

CONFIRMED matched warm baseline: mean 195.1ms, median 196.2ms, p95 325.3ms.
CONFIRMED matched warm candidate: mean 276.8ms, median 262.0ms, p95 397.7ms.

INFERRED snapshot interaction can tolerate the measured approximately .28s mean inference, but the added cost is unwarranted given worse accuracy and confidence safety. This 40-image equal-route benchmark is not live-webcam end-to-end latency or a validation-frequency-weighted average. Earlier 73ms hybrid timing was a different run; use the matched baseline measured here for incremental cost.

FAIL on all three admission criteria: fewer correct answers, wrong-at-95% above 39, and nine previously >=95% classes regress. MediaPipe fallback is retained, but a confidently wrong RTMPose proposal preempts it. High classifier probability is insufficient evidence that a two-hand proposal should override MP.

UNVERIFIED new-person and real-webcam accuracy. No further variant was selected from validation.

## Reproduction

Run from project root with the project venv. tune_detector.py writes the frozen training policy; evaluate_ensemble.py uses it without tuning. The grid took several minutes; full evaluation also took several minutes. Both commands log output and results under this directory.

    ./venv/Scripts/python.exe tracker_v2/hybrid_candidate/double_ensemble/tune_detector.py > tracker_v2/hybrid_candidate/double_ensemble/grid.log 2>&1
    ./venv/Scripts/python.exe tracker_v2/hybrid_candidate/double_ensemble/evaluate_ensemble.py > tracker_v2/hybrid_candidate/double_ensemble/evaluation.log 2>&1

Runtime usage from the existing staging module: from candidate import Hybrid; engine = Hybrid(double_ensemble=True); engine.predict(rgb_image). Output preserves the hybrid dictionary and adds double_source on double routes.

All original production runtime hashes were checked before/after evaluation. Sources, grid policy, runtime asset hashes, row predictions and per-class outcomes are retained here.
