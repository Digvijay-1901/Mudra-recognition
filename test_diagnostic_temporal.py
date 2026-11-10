"""Acceptance checks for Diagnostic-only smoothing and the folder validator."""
import sys
sys.dont_write_bytecode=True
import csv
import io
import json
import tempfile
import hashlib
import importlib.util
from datetime import datetime
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from streamlit.testing.v1 import AppTest
import brain
from diagnostic_temporal import smooth_capture
from validate_hybrid_live import run

ROOT=Path(__file__).resolve().parent

def vote(state,t,label="Pathaka",confidence=.99,mode="Single",expected="Pathaka"):
    return smooth_capture(state,expected,"auto",.3,t,label,confidence,mode)

def rules():
    state={}
    assert not vote(state,0,"Chakra",.7,"Double")["changed"]
    state={}
    vote(state,0);vote(state,3)
    assert vote(state,6,"Chakra",.7,"Double")["label"]=="Pathaka"
    assert vote(state,9,"Chakra",.7,"Double")["label"]=="Chakra"
    assert len(state["history"])==4
    assert not vote(state,40,"Chakra",.7,"Double")["changed"]
    state={}
    vote(state,0);vote(state,3)
    assert not vote(state,4,"Chakra",.7,"Double",expected="Ardhachandra")["changed"]
    state={}
    vote(state,0);vote(state,3)
    assert not vote(state,5,"Chakra",.99,"Double")["changed"]
    assert not vote(state,6,"NO_HAND",None,"Unknown")["changed"]
    state={}
    vote(state,0,expected="Other");vote(state,3,expected="Other")
    assert vote(state,6,"Chakra",.7,"Double",expected="Other")["label"]=="Pathaka"
    print("PASS isolated, majority, persistent change, expiry, context, high confidence, no-hand and no expected-label vote")

def historical():
    state={};counts={c:dict(errors=0,corrected=0,harmed=0) for c in ("Pathaka","Ardhachandra")}
    telemetry=ROOT/"reports/webcam_diagnostics.csv"
    if not telemetry.is_file():
        print("SKIP historical webcam replay: reports/webcam_diagnostics.csv is absent")
        return None
    with telemetry.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        c=r["Expected_Class"]
        confidence=float(r["Confidence"]) if r["Confidence"] else None
        result=smooth_capture(state,c,"auto",.3,datetime.fromisoformat(r["Timestamp"]).timestamp(),
                              r["Predicted_Class"],confidence,r["Mode_Detected"])
        if c in counts:
            wrong=r["Predicted_Class"]!=c
            counts[c]["errors"]+=int(wrong)
            counts[c]["corrected"]+=int(wrong and result["label"]==c)
            counts[c]["harmed"]+=int(not wrong and result["label"]!=c)
    print("CONFIRMED historical raw telemetry replay",json.dumps(counts))
    return counts

def analyzer_smoke(folder,frames,telemetry):
    from analyze_webcam_frames import run as analyze, mediapipe_only, prediction_fields, original_points
    source=cv2.imread(str(ROOT/"examples/pathaka.jpg"))
    for name,image in [("rotated.png",cv2.rotate(source,cv2.ROTATE_90_CLOCKWISE)),
                       ("blank.png",np.zeros((256,256,3),np.uint8)),
                       ("two.png",cv2.imread(str(ROOT/"examples/anjali_two.jpg")))]:
        ok,png=cv2.imencode(".png",image);assert ok
        (frames/name).write_bytes(png.tobytes())
    inputs=[*frames.glob("*.png"),telemetry]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows=analyze(frames,folder/"overlays",telemetry)
    assert len(rows)==9 and not any(r["Error"] for r in rows)
    assert all(Path(r["Overlay"]).is_file() for r in rows)
    assert len(list((folder/"overlays").glob("*.png")))==9
    blank=next(r for r in rows if Path(r["Frame"]).name=="blank.png")
    assert blank["Hybrid_Label"]==blank["MP_Label"]=="NO_HAND"
    assert blank["RTMPose_Hands"]==blank["MediaPipe_Hands"]==0
    assert next(r for r in rows if Path(r["Frame"]).name.startswith("2026"))["Expected"]=="Pathaka"
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    mapped=original_points([np.array([[.5,.5,0.]])],(16,32,3),(10,20,3))
    assert np.array_equal(mapped[0],[[10.,5.]])
    # Independent reference: actual pre-promotion source, not rewritten test logic.
    previous=ROOT/"tests/fixtures/brain.before.py"
    assert previous.is_file(),"Pre-promotion source required for MP parity acceptance"
    spec=importlib.util.spec_from_file_location("_old_webcam_brain",previous)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    old.BASE=str(ROOT)
    for image in [source,cv2.rotate(source,cv2.ROTATE_90_CLOCKWISE),
                  cv2.imread(str(ROOT/"examples/anjali_two.jpg")),np.zeros((256,256,3),np.uint8)]:
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        a=old.predict(rgb,threshold=.30);b=mediapipe_only(rgb)
        label_a="NO_HAND" if a is None else "ABSTAIN" if isinstance(a,old.Abstention) else a.label
        assert label_a==prediction_fields(b)["label"]
        if a is not None and not isinstance(a,old.Abstention):
            assert np.isclose(a.confidence,b.confidence,atol=1e-12,rtol=0)
    print("PASS analyzer nine PNGs/overlays, blank/rotation/two-hand cases, padding transform, input immutability and old-source MP parity 4/4")


def main():
    rules()
    protected=[ROOT/name for name in ("brain.py","feature_extraction.py","detector.py")]
    manifest=json.loads((ROOT/"tracker_v2/hybrid_candidate/manifest.json").read_text())
    protected += [ROOT/name for name in manifest["runtime_inputs"] if name.endswith((".joblib",".onnx"))]
    protected_before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    with tempfile.TemporaryDirectory(prefix="mudra-temporal-") as directory, threadpool_limits(limits=1):
        folder=Path(directory)
        payload=(ROOT/"examples/pathaka.jpg").read_bytes()
        image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
        from gate import _model
        _model().n_jobs=1
        actual=brain.predict(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),threshold=.30)
        assert actual is not None and not isinstance(actual,brain.Abstention)
        pathaka=replace(actual,label="Pathaka",confidence=.99,mode="Single-hand")
        chakra=replace(actual,label="Chakra",confidence=.7,mode="Double-hand")
        telemetry=folder/"diagnostics.csv"
        frames=folder/"frames"
        # Exercise the existing non-destructive schema rollover as fields grow.
        with telemetry.open("w",newline="",encoding="utf8") as handle:
            writer=csv.writer(handle)
            writer.writerow(["Timestamp","Expected_Class","Predicted_Class","Confidence","Mode_Detected","Hands_Count","Tilt_Degrees"])
            writer.writerow(["2026-11-08T08:00:00+05:30","Pathaka","Pathaka",.99,"Single",1,0])
        with patch("app_telemetry.DIAGNOSTICS_CSV",telemetry), patch("app_telemetry.DIAGNOSTIC_FRAMES",frames):
            source=(ROOT/"streamlit_app.py").read_text(encoding="utf8")
            app=AppTest.from_string(source,default_timeout=60).run()
            assert not app.exception,[e.message for e in app.exception]
            next(x for x in app.sidebar.checkbox if x.label=="Save frames for debugging").set_value(True).run()
            assert not app.exception,[e.message for e in app.exception]
            archived=list(folder.glob("diagnostics.archive-*.csv"))
            assert len(archived)==1
            with archived[0].open(newline="",encoding="utf8") as handle:
                assert len(list(csv.DictReader(handle)))==1
            def run_webcam():
                # AppTest does not serialize stateful tab/expander blocks yet.
                # Supply the same state the browser sends with each rerun.
                app.session_state["image_source"]="Webcam"
                app.session_state["diagnostic_panel"]=True
                return app.run()
            run_webcam()
            for i,pred in enumerate([pathaka,pathaka,pathaka,chakra]):
                app.session_state["diagnostic_sequence"]=i+1
                app.session_state["diagnostic_context"]=dict(sequence=i+1,expected="Pathaka",mode="auto",threshold=.3,
                                                            timestamp=f"2026-11-09T10:00:{i*3:02d}+05:30",captured_at=i*3)
                with patch("streamlit.camera_input",return_value=io.BytesIO(payload)),patch("brain.predict",return_value=pred) as call:
                    run_webcam();run_webcam()
                    assert call.call_count==1
                assert not app.exception,[e.message for e in app.exception]
            record=app.session_state["diagnostic_attempt"]
            assert record["temporal"]["label"]=="Pathaka" and record["label"]=="Chakra"
            assert len(app.session_state["diagnostic_temporal"]["history"])==4
            with telemetry.open(newline="") as f:
                logged=list(csv.DictReader(f))
            assert len(logged)==4 and logged[-1]["Predicted_Class"]=="Chakra"
            assert logged[-1]["Displayed_Label"]=="Pathaka" and logged[-1]["Message"]==""
            assert len(list(frames.glob("*.png")))==4
            for row in logged:
                saved=cv2.imdecode(np.fromfile(row["Frame_Path"],dtype=np.uint8),cv2.IMREAD_COLOR)
                assert np.array_equal(saved,image), "Saved frame is not the unannotated camera pixels"
            message="Only one hand detected. Show both hands for a two-hand mudra, or choose Single-hand mode if appropriate."
            abstain=brain.Abstention(message,actual.results,actual.used_image,actual.tilt_degrees)
            for i,pred in enumerate([abstain,None],5):
                app.session_state["diagnostic_sequence"]=i
                app.session_state["diagnostic_context"]=dict(sequence=i,expected="Pathaka",mode="auto",threshold=.3,
                                                            timestamp="2026-11-09T10:00:09+05:30",captured_at=i*3)
                with patch("streamlit.camera_input",return_value=io.BytesIO(payload)),patch("brain.predict",return_value=pred) as call:
                    run_webcam();run_webcam()
                    assert call.call_count==1
                assert not app.exception,[e.message for e in app.exception]
            with telemetry.open(newline="",encoding="utf8") as handle:
                extra=list(csv.DictReader(handle))
            assert len(extra)==6 and len(list(frames.glob("*.png")))==6
            assert extra[-2]["Message"]==message and extra[-2]["Displayed_Label"]=="ABSTAIN"
            assert extra[-1]["Predicted_Class"]=="NO_HAND" and "No hand detected" in extra[-1]["Message"]
            assert len({r["Frame_Path"] for r in extra})==6
            print("PASS lossless raw PNGs, one save per capture, filename collisions, schema archive, displayed label, abstain/no-hand messages")
            next(x for x in app.selectbox if x.label=="Expected mudra").set_value("Ardhachandra").run()
            assert not app.exception
            assert "diagnostic_temporal" not in app.session_state
            print("PASS real app startup, four reconstructed captures, duplicate reruns, raw logging, displayed smoothing and selection reset")
        photos=folder/"photos"/"Pathaka"
        photos.mkdir(parents=True)
        import os
        for i in range(4):
            p=photos/f"{i}.jpg";p.write_bytes(payload);os.utime(p,(1000+3*i,1000+3*i))
        with patch("brain.predict",side_effect=[pathaka,pathaka,pathaka,chakra]):
            result=run(photos.parent,folder/"sequence",True)
        assert result["raw_accuracy"]==.75 and result["displayed_accuracy"]==1
        with patch("brain.predict",side_effect=[pathaka,pathaka,pathaka,chakra]):
            result=run(photos.parent,folder/"independent",False)
        assert result["raw_accuracy"]==result["displayed_accuracy"]==.75
        import hybrid_single
        with patch("hybrid_single.classify", wraps=hybrid_single.classify) as rtmpose:
            result=run(photos.parent,folder/"real-inference",False)
        assert rtmpose.call_count==4, "Validator did not use production RTMPose single path"
        assert result["errors"]==0 and result["images"]==4
        print("PASS validator replay, independent isolation, CSV/JSON export and real production inference")
        analyzer_smoke(folder,frames,telemetry)
    historical()
    assert protected_before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    print("PASS protected inference sources, model files and ONNX assets unchanged")
    print("ACCEPTANCE PASS")

if __name__=="__main__":
    main()
