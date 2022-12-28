import os
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort


def generate_feat(model, imgdir):
    # mediapipe face detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection =  mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.8).process

    # set model for generate feature vector
    input_name = model.get_inputs()[0].name
    # save data
    _LABEL2NAME = {}
    idx = 0
    feats = []

    imgs = os.listdir(imgdir)
    for img in imgs:
        img_name = img
        img_path = os.path.join(imgdir, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection(img)
        if results.detections is None:
            print(f"Not detect face : {img_path}")
            continue
        for detection in results.detections:
            location = detection.location_data
            if not location.HasField('relative_bounding_box'):
                print(f"losing location info : {img_path}")
                continue
            relative_bounding_box = location.relative_bounding_box
            rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                relative_bounding_box.xmin, 
                relative_bounding_box.ymin, 
                img.shape[1],
                img.shape[0])
                
            if rect_start_point is None:
                print(f"losing start point : {img_path}")
                continue
            rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height, 
                img.shape[1],
                img.shape[0])
                
            if rect_end_point is None:
                print(f"losing end point : {img_path}")
                continue
            face = cv2.resize(
                    img[
                        rect_start_point[1]:rect_end_point[1], 
                        rect_start_point[0]:rect_end_point[0]
                        ], 
                        (112, 112))
            
            face = ((face / 255. - 0.5) / 0.5)[None,...].transpose((0, 3, 1, 2))
            out = model.run(None, {input_name: face.astype(np.float32)})[0][0]
            feats.append(out)
            _LABEL2NAME[str(idx)] = img_name[:-4]
            idx += 1
    feats = np.array(feats)
    
    return feats, _LABEL2NAME