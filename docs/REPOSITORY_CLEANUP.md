# CONFIRMED cleanup dependency inventory

Read-only reference scan of production entry points and tracked tests. No model deserialization.

| Initially untracked file | Direct import/open evidence | Decision |
| --- | --- | --- |
| confusion_audit.py | No production/test import or open found | IGNORE |
| mirror_hms_audit.py | No production/test import or open found | IGNORE |
| pinky_pair_audit.py | No production/test import or open found | IGNORE |
| reports/double_head_promotion_20261108T071737072492Z/verify.py | No production/test import or open found | IGNORE |
| reports/hybrid_live_wiring/brain.before.py | test_diagnostic_temporal.py:87: previous=ROOT/"reports/hybrid_live_wiring/brain.before.py" | Archive byte-identical test fixture; leave required reports path ignored on disk; restore command documented |
| reports/hybrid_live_wiring/streamlit_app.before.py | No production/test import or open found | IGNORE |
| reports/tasks_handlandmarker_check/binary.py | No production/test import or open found | IGNORE |
| reports/tasks_handlandmarker_check/compare.py | No production/test import or open found | IGNORE |
| routing_calibration_review.py | No production/test import or open found | IGNORE |
| test_tilt_schema.py | No production/test import or open found | IGNORE |
| tracker_v2/README.md | No production/test import or open found | IGNORE |
| tracker_v2/build_features.py | No production/test import or open found | IGNORE |
| tracker_v2/common.py | No production/test import or open found | IGNORE |
| tracker_v2/detection_report.json | No production/test import or open found | IGNORE |
| tracker_v2/evaluate.py | No production/test import or open found | IGNORE |
| tracker_v2/evaluation.json | No production/test import or open found | IGNORE |
| tracker_v2/evaluation_summary.md | No production/test import or open found | IGNORE |
| tracker_v2/extract_landmarks.py | No production/test import or open found | IGNORE |
| tracker_v2/extraction_manifest.json | No production/test import or open found | IGNORE |
| tracker_v2/features_manifest.json | test_hybrid_production.py:57: meta = json.loads((ROOT / "tracker_v2/features_manifest.json").read_text()) | TRACK explicit filename |
| tracker_v2/geometry_provenance.json | No production/test import or open found | IGNORE |
| tracker_v2/hybrid_candidate/README.md | No production/test import or open found | IGNORE |
| tracker_v2/hybrid_candidate/evaluate_hybrid.py | No production/test import or open found | IGNORE |
| tracker_v2/hybrid_candidate/manifest.json | test_hybrid_production.py:53: manifest = json.loads((STAGE / "manifest.json").read_text())<br>test_hybrid_production.py:57: meta = json.loads((ROOT / "tracker_v2/features_manifest.json").read_text())<br>test_diagnostic_temporal.py:106: manifest=json.loads((ROOT/"tracker_v2/hybrid_candidate/manifest.json").read_text()) | TRACK explicit filename |
| tracker_v2/hybrid_candidate/protected_before.json | No production/test import or open found | IGNORE |
| tracker_v2/hybrid_candidate/results.json | test_hybrid_production.py:112: previous = json.loads((STAGE / "results.json").read_text()) | TRACK explicit filename |
| tracker_v2/hybrid_candidate/summary.md | No production/test import or open found | IGNORE |
| tracker_v2/policy_v2.json | No production/test import or open found | IGNORE |
| tracker_v2/protected_before.json | No production/test import or open found | IGNORE |
| tracker_v2/requirements.txt | No production/test import or open found | TRACK explicit filename |
| tracker_v2/setup_check.py | No production/test import or open found | IGNORE |
| tracker_v2/setup_report.json | No production/test import or open found | IGNORE |
| tracker_v2/train_models.py | No production/test import or open found | IGNORE |
| tracker_v2/training_manifest.json | No production/test import or open found | IGNORE |
| tracker_v2/verification.json | No production/test import or open found | IGNORE |
| train_v4.py | No production/test import or open found | IGNORE |
| validation_trials.py | No production/test import or open found | IGNORE |
| webcam_audit.py | No production/test import or open found | IGNORE |

Grep used for direct dependencies:
```powershell
rg -n "brain.before.py|features_manifest.json|manifest.json|results.json" test_hybrid_production.py test_diagnostic_temporal.py
rg -n "model_single|rtmdet_hand|rtmpose_hand5|from tracker_v2" hybrid_single.py tracker_v2/adapter.py
```

tracker_v2/requirements.txt is retained because docs/RUNTIME_ASSETS.md uses it to install the RTMPose runtime; it is not imported as Python.

All other initially untracked scripts/metadata above are ignored by explicit path; no broad git add is used.
