# Webcam Tripathaka diagnosis (read-only)

CONFIRMED: no capture, preprocessing, inference, model or training files changed. Decision: no fix; retain production behavior.

## A-C: transforms and pixel layout

| Input, 20 Tripathaka frames | Hybrid correct | MediaPipe correct | Hybrid wrong >=90% | MP wrong >=90% |
| --- | --- | --- | --- | --- |
| original | 1/20 | 1/20 | 19 | 18 |
| horizontal_flip | 0/20 | 4/20 | 19 | 15 |
| wrong_BGR | 0/20 | 1/20 | 6 | 7 |
| highres_resize | 0/20 | 3/20 | 17 | 16 |
| highres_fit | 0/20 | 1/20 | 17 | 18 |
| typical_resize | 0/20 | 0/20 | 5 | 12 |
| typical_fit | 0/20 | 0/20 | 15 | 16 |

CONFIRMED: actual app preprocessing function vs actual validator file loop: identical RGB pixel arrays, shapes and dtype on 30/30 PNGs. Both OpenCV-decode BGR then convert BGR->RGB exactly once; neither path flips/resizes. Wrong_BGR is a deliberate invalid-input control, not a proposed change.

CONFIRMED: 613x344 once, 613x459 29 times. Representative outside Tripathaka size is 218x212; stretch and aspect-preserving fit both yield hybrid/MP 0/20. High-resolution control also does not rescue hybrid.

Outside photo sizes (width,height, count): {"(1792, 2400)": 3, "(210, 213)": 1, "(210, 211)": 1, "(216, 206)": 1, "(322, 215)": 1, "(317, 210)": 1, "(215, 211)": 1, "(208, 206)": 1, "(211, 213)": 1, "(316, 217)": 1, "(318, 210)": 1, "(1024, 746)": 1, "(1278, 1230)": 4, "(236, 307)": 1, "(230, 300)": 1, "(230, 307)": 2, "(162, 210)": 1, "(151, 207)": 1, "(160, 212)": 1, "(152, 207)": 1, "(156, 213)": 1, "(160, 206)": 1, "(156, 212)": 1, "(158, 215)": 1, "(1536, 1024)": 1, "(218, 212)": 1, "(213, 217)": 1, "(317, 207)": 1, "(318, 207)": 1, "(207, 222)": 1, "(212, 217)": 1, "(211, 221)": 1, "(317, 202)": 1, "(317, 197)": 1}

All ten Tripathaka outside photos have unique sizes. Initial 1536x1024 modal-tie selection was a high-resolution control. Representative 218x212 is the actual photo size nearest the median log width/height; selected without considering prediction success.

## D: ring-finger training comparison

CONFIRMED: only existing clean TRAINING partition features used. Distance = norm(tip-wrist) / maximum wrist-to-any-landmark distance, in pixel-scaled XYZ. This is existing feature 66; five tips occupy features 63-67. Smaller means more folded. RT Z=0; MP Z retained. Statistics compare each tracker with its own cached raw features.

| Tracker | Class | n | min | p05 | median | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rtmpose | Tripathaka | 438 | 0.131 | 0.405 | 0.771 | 0.957 | 1.000 |
| mediapipe | Tripathaka | 396 | 0.374 | 0.472 | 0.684 | 0.901 | 1.000 |
| rtmpose | Pathaka | 543 | 0.846 | 0.912 | 0.940 | 0.980 | 0.993 |
| mediapipe | Pathaka | 466 | 0.706 | 0.840 | 0.910 | 0.950 | 0.991 |

CONFIRMED live ring summaries and range membership: {"rtmpose": {"live_min": 0.6869974136352539, "live_median": 0.9500169456005096, "live_max": 0.9672448039054871, "central90_categories": {"Tripathaka": 4, "Pathaka": 7, "overlap": 9}, "full_range_categories": {"Tripathaka": 2, "overlap": 18}}, "mediapipe": {"live_min": 0.7866175770759583, "live_median": 0.9488558769226074, "live_max": 0.9667023420333862, "central90_categories": {"Pathaka": 8, "outside_both_central_ranges": 9, "overlap": 2, "Tripathaka": 1}, "full_range_categories": {"overlap": 20}}}

CONFIRMED: live normalized ring median RT=0.9500, MP=0.9489. Original training medians Tripathaka/Pathaka RT=.7708/.9399, MP=.6842/.9104. Full ranges overlap for 18/20 RT and 20/20 MP live values; no scalar proves the class. INFERRED: the ring is represented too extended in many live frames. Tracker/visual ambiguity versus model boundary weakness remains UNVERIFIED without ground-truth landmarks.

| Frame | RT thumb | index | middle | ring | little | MP thumb | index | middle | ring | little | RT central90 category | MP central90 category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.652 | 0.966 | 1.000 | 0.687 | 0.815 | 0.644 | 0.952 | 1.000 | 0.913 | 0.842 | Tripathaka | Pathaka |
| 2 | 0.653 | 0.958 | 1.000 | 0.846 | 0.747 | 0.650 | 0.965 | 1.000 | 0.958 | 0.852 | Tripathaka | outside_both_central_ranges |
| 3 | 0.675 | 0.958 | 1.000 | 0.962 | 0.824 | 0.704 | 0.937 | 1.000 | 0.957 | 0.836 | Pathaka | outside_both_central_ranges |
| 4 | 0.650 | 0.962 | 1.000 | 0.948 | 0.820 | 0.658 | 0.957 | 1.000 | 0.933 | 0.829 | overlap | Pathaka |
| 5 | 0.646 | 0.959 | 1.000 | 0.963 | 0.824 | 0.660 | 0.952 | 1.000 | 0.927 | 0.835 | Pathaka | Pathaka |
| 6 | 0.649 | 0.957 | 1.000 | 0.955 | 0.821 | 0.647 | 0.953 | 1.000 | 0.928 | 0.827 | overlap | Pathaka |
| 7 | 0.644 | 0.959 | 1.000 | 0.876 | 0.777 | 0.664 | 0.951 | 1.000 | 0.873 | 0.814 | Tripathaka | overlap |
| 8 | 0.645 | 0.955 | 1.000 | 0.927 | 0.799 | 0.651 | 0.967 | 1.000 | 0.869 | 0.798 | overlap | overlap |
| 9 | 0.646 | 0.970 | 1.000 | 0.898 | 0.780 | 0.665 | 0.972 | 1.000 | 0.924 | 0.818 | Tripathaka | Pathaka |
| 10 | 0.643 | 0.965 | 1.000 | 0.965 | 0.815 | 0.682 | 0.949 | 1.000 | 0.949 | 0.813 | Pathaka | Pathaka |
| 11 | 0.654 | 0.971 | 1.000 | 0.920 | 0.800 | 0.664 | 0.960 | 1.000 | 0.787 | 0.811 | overlap | Tripathaka |
| 12 | 0.636 | 0.960 | 1.000 | 0.949 | 0.815 | 0.657 | 0.951 | 1.000 | 0.967 | 0.835 | overlap | outside_both_central_ranges |
| 13 | 0.642 | 0.957 | 1.000 | 0.958 | 0.814 | 0.635 | 0.955 | 1.000 | 0.958 | 0.823 | Pathaka | outside_both_central_ranges |
| 14 | 0.643 | 0.959 | 1.000 | 0.959 | 0.815 | 0.654 | 0.950 | 1.000 | 0.943 | 0.825 | Pathaka | Pathaka |
| 15 | 0.631 | 0.960 | 1.000 | 0.967 | 0.820 | 0.646 | 0.963 | 1.000 | 0.957 | 0.818 | Pathaka | outside_both_central_ranges |
| 16 | 0.619 | 0.955 | 1.000 | 0.913 | 0.796 | 0.648 | 0.933 | 1.000 | 0.965 | 0.803 | overlap | outside_both_central_ranges |
| 17 | 0.651 | 0.958 | 1.000 | 0.945 | 0.815 | 0.663 | 0.954 | 1.000 | 0.949 | 0.822 | overlap | Pathaka |
| 18 | 0.653 | 0.957 | 1.000 | 0.951 | 0.806 | 0.643 | 0.943 | 1.000 | 0.952 | 0.809 | overlap | outside_both_central_ranges |
| 19 | 0.659 | 0.961 | 1.000 | 0.952 | 0.806 | 0.639 | 0.950 | 1.000 | 0.954 | 0.815 | overlap | outside_both_central_ranges |
| 20 | 0.646 | 0.963 | 1.000 | 0.959 | 0.823 | 0.646 | 0.936 | 1.000 | 0.963 | 0.820 | Pathaka | outside_both_central_ranges |

Per-frame predictions, modes and confidence for every transform are in diagnosis.json. Full arrays/training images were never printed.

## E: handedness

CONFIRMED: live MP Right 20/20; original detected training Tripathaka Left 304, Right 92 (76.8% opposite MP orientation). INFERRED physical Left under stock Streamlit unmirrored capture: local CameraInput default mirrored=false and MP docs assume mirrored inputs. Camera-driver mirroring and absolute physical handedness are UNVERIFIED. Flip failure argues against a simple handedness preprocessing fix.

Local convention sources read: venv/Lib/site-packages/mediapipe/python/solutions/hands.py (mirrored-hand convention) and streamlit/static/static/js/CameraInput.BGpQNOjY.js (mirrored:false default).

## Acceptance rerun

| Check | Result |
| --- | --- |
| Clean validation, actual production API on cached detector records | 90.55% accuracy / 93.24% coverage / 39 wrong-at-95%; 4465/4465 staged decisions match |
| Fresh source-pixel validation | 20/20 staged matches |
| Outside validator | 38/40; Pathaka 17/17, Mayura 12/13, Tripathaka 9/10 |
| Live before -> after no-change rerun | Tripathaka 1/20 -> 1/20; Pathaka 10/10 -> 10/10; 30/30 decisions identical |
| Existing tests | ACCEPTANCE PASS; protected source/model/asset hashes unchanged |

INFERRED conclusion: strongest evidence is D, a ring-keypoint/visual/generalization weakness; A-C did not reveal a mismatch. E is an orientation imbalance, not a proven cause. No evidence justifies changing preprocessing or tuning on these captures.
