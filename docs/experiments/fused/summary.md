# Completed fused smoke investigation

CONFIRMED outcome: negative smoke result. Stage 4 full workflow is not warranted.

## Worker and checkpoint diagnosis
The gate RandomForest used joblib threading, whose ThreadPool bookkeeping creates a multiprocessing SimpleQueue backed by a Windows named pipe. WinError 5 occurred at _winapi.CreateFile while opening that pipe, before pool initialization completed. The observed cause is OS permission denial for the worker-pool IPC object; which external policy denied it remains UNVERIFIED. The existing staging adapter sets the cached gate n_jobs=1 before all predictions, avoiding the worker pool entirely. This does not change model weights or decision policy.
The subsequent adapter change invalidated the checkpoint signature by design. Removed the entire authorized one-record smoke directory after verifying its absolute path. The stable adapter then completed a fresh 80-image run from zero, including all ten atomic checkpoints. No old record was reused. The run took about 44 seconds excluding setup. Source manifests and photos were read only to perform extraction/checkpoint verification.
This smoke uses the original deterministic balanced-class sampler over source records (60 single, 20 double), not a new held-out accuracy evaluation. No classifier was selected or trained from its results.

## Matched fresh detection counts
| Pipeline | Single any /60 | Double any /20 | Double two /20 |
| --- | --- | --- | --- |
| MediaPipe | 54 | 20 | 4 |
| RTMPose | 59 | 12 | 2 |
| Current hybrid | 54 | 20 | 4 |
| Fused | 45 | 12 | 5 |

Single two-hand false outputs: MP 0, RTMPose 1, hybrid 0, fused 0.
Counts refer to accepted landmark outputs, not verified distinct true hands. Hybrid hand count describes the tracker selected by its gate, independent of label acceptance; gate abstention retains MP hand count.
The old extractor's generic console headers call fused output RTMPose; smoke_comparison.json and this matched table provide the explicit correct arm names.

## Forty-image visual review
CONFIRMED subjective review of 20 single-hand photos and all 20 double-hand photos. Reasonable 9/40; oversized 8/40; overlapping two-hand regions 6/40; missing/incomplete 17/40. Missing/incomplete includes failure to box the second visible hand. Categories are mutually exclusive descriptions, not ground-truth localization scores. Overlap can be legitimate for overlapping physical hands. Cyan joints also reveal that accepted output count can exceed valid distinct-hand localization.
Individual image IDs and assignments are in quality_review.json; four quality_*.jpg sheets preserve evidence.

INFERRED: the unrotated enclosing-box interface loses the benefit of MediaPipe's oriented ROI, often exposes multiple hands to a single-hand pose head, and the standalone no-retry palm path loses coverage relative to the full detector wrapper. These are plausible explanations, not isolated causal measurements.
The extra double two-output case does not offset reduced single/any-hand coverage or establish classification improvement. Full Auto accuracy versus 90.55% remains UNVERIFIED. Stop at this negative smoke gate. No full extraction, training, evaluation or production promotion.
Production runtime input hashes verified unchanged.
