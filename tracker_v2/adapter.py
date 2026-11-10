"""RTMPose-Hand5 CPU adapter. No MediaPipe fallback or label-aware box hints."""
import os
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
MODEL_FILES={
 'detector':('rtmdet_hand.onnx','https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip'),
 'pose':('rtmpose_hand5.onnx','https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip')}

class Tracker:
    def __init__(self):
        # Limit ORT threads: the default can be slower for these small CPU graphs.
        import onnxruntime as ort
        from rtmlib import Hand
        original=ort.InferenceSession
        def session(*args,**kwargs):
            options=ort.SessionOptions();options.intra_op_num_threads=min(4,os.cpu_count() or 1);options.inter_op_num_threads=1
            kwargs.setdefault('sess_options',options)
            return original(*args,**kwargs)
        ort.InferenceSession=session
        try:self.model=Hand(det=str(BASE/'assets/rtmdet_hand.onnx'),pose=str(BASE/'assets/rtmpose_hand5.onnx'),backend='onnxruntime',device='cpu',to_openpose=False)
        finally:ort.InferenceSession=original

    def detect(self,image):
        import cv2
        h,w=image.shape[:2]
        for attempt in range(3):
            used=image
            if attempt==1:used=cv2.copyMakeBorder(image,h//3,h//3,w//3,w//3,cv2.BORDER_REPLICATE)
            if attempt==2:used=cv2.copyMakeBorder(image,h//6,h//6,w//6,w//6,cv2.BORDER_REFLECT_101)
            # Hand.__call__ treats an empty bbox list as whole-image pose in some
            # versions. Explicitly skip pose when detector has no boxes.
            boxes=self.model.det_model(used)
            if len(boxes)==0:continue
            points,scores=self.model.pose_model(used,bboxes=boxes)
            candidates=[]
            for p,s in zip(points,scores):
                if p.shape!=(21,2) or not np.isfinite(p).all() or not np.isfinite(s).all():continue
                if np.mean(s)<.3 or min(s[0],s[9])<.3:continue
                if np.linalg.norm(p[9]-p[0])<1e-3:continue
                q=np.zeros((21,3),dtype=np.float32)
                q[:,:2]=p/np.array([used.shape[1],used.shape[0]])
                candidates.append((float(np.mean(s)),q,np.asarray(s,dtype=np.float32)))
            if candidates:
                # Same two-hand maximum as legacy; no true class/count supplied.
                candidates=sorted(candidates,key=lambda x:x[0],reverse=True)[:2]
                candidates=sorted(candidates,key=lambda x:float(x[1][0,0]))
                return [x[1] for x in candidates],[x[2] for x in candidates],used
        return [],[],image
