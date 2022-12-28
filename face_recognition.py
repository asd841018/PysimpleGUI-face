"""
    Demo of Face recognition with a WebCam using PySimpleGUI
    Reference : https://github.com/PySimpleGUI/PySimpleGUI-YOLO
                https://github.com/deepinsight/insightface
                https://github.com/kprokofi/light-weight-face-anti-spoofing
"""
from pathlib import Path
cwd = Path(__file__).parent.absolute()  ## this folder's absolute path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import PySimpleGUI as sg
import mediapipe as mp
import onnxruntime as ort
import pickle
import ujson as json
import yaml
from utils.functions import cos_dist, most_common
from utils.generate_data import generate_feat

sg.theme('LightGreen')

config_path = cwd / 'config.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

FAS_model = ort.InferenceSession(config['fas_model'], providers=['CUDAExecutionProvider'])
FAS_input_name = FAS_model.get_inputs()[0].name

model = ort.InferenceSession(config['model'], providers=['CUDAExecutionProvider'])
input_name = model.get_inputs()[0].name

feats, _LABEL2NAME = generate_feat(model, config['img_dir'])


# with open('./probe/label_dict.json', 'r') as f:
#     _LABEL2NAME = json.load(f)

# with open('./probe/DP_res100.pkl', 'rb') as file:
#     feats = pickle.load(file)

simi_threshold = 0.5      # initial settings
fas_threshold = 0.8      # initial settings
camera_number = 0       # if you have more than 1 camera, change this variable to choose which is used
count = 0    # for counting how much face is passed
simi_list = []
label_list = []

sg.popup_quick_message('Loading mediapipe face detection...', background_color='blue', text_color='white')

# set face detection with mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# initialize the video stream, pointer to output video file, and
# frame dimensions
W, H = None, None
win_started = False
cap = cv2.VideoCapture(camera_number)  # initialize the capture device
with mp_face_detection.FaceDetection(model_selection=0, 
                                     min_detection_confidence=0.8,
                                     ) as face_detection:
    while True:
        # read the next frame from the file or webcam
        grabbed, frame = cap.read()

        # if the frame was not grabbed, then we stream has stopped so break out
        if not grabbed:
            continue

        # if the frame dimensions are empty, grab them
        if not W or not H:
            (H, W) = frame.shape[:2]  #(H, W ,C)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame)
        im_frame = Image.fromarray(frame)
        if results.detections:
            draw = ImageDraw.Draw(im_frame)
            for detection in results.detections:
                location = detection.location_data
                if not location.HasField('relative_bounding_box'):
                    continue
                relative_bounding_box = location.relative_bounding_box
                rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, 
                    relative_bounding_box.ymin, 
                    W,
                    H
                )

                if rect_start_point is None:
                    continue
                rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, 
                    W,
                    H
                )

                if rect_end_point is None:
                    continue

                # if count == 10:
                #     avg_simi = np.mean(simi_list)
                #     most_idx = most_common(label_list)
                #     if avg_simi > simi_threshold:
                #         draw.rectangle(
                #                 [rect_start_point, rect_end_point], outline=(0, 255, 0), width=6
                #             )

                #         draw.text(
                #                     (rect_start_point[0], rect_start_point[1]-10), 
                #                     text=f"{str(_LABEL2NAME[str(most_idx)])}_ {avg_simi:.3f}",
                #                 )
                #     else:
                #         draw.rectangle(
                #                 [rect_start_point, rect_end_point], outline=(255, 255, 255), width=6
                #             )

                #         draw.text(
                #                     (rect_start_point[0], rect_start_point[1]-10), 
                #                     text='unknown',
                #                 )

                # else:
                face = cv2.resize(
                            frame[
                                rect_start_point[1]:rect_end_point[1], 
                                rect_start_point[0]:rect_end_point[0]
                            ], 
                            (112, 112),
                        )
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face =  ((face / 255. - 0.5) / 0.5)[None,...].transpose((0, 3, 1, 2))
                # model inference
                out = FAS_model.run(None, {FAS_input_name: face.astype(np.float32)})[0][0]
                prob = np.max(out)
                idx = np.argmax(out)
                print(fas_threshold)
                if idx == 0 and prob > fas_threshold:
                    out = model.run(None, {'input': face.astype(np.float32)})[0]
                    dist = cos_dist(f1=feats, f2=out)
                    simi = np.max(dist)
                    label = int(np.argmax(dist))
                    
                    if simi > simi_threshold:
                        draw.rectangle(
                            [rect_start_point, rect_end_point], outline=(0, 255, 0), width=6
                        )

                        draw.text(
                                    (rect_start_point[0], rect_start_point[1]-10), 
                                    text=f"{str(_LABEL2NAME[str(label)])}_{simi:.3f}",
                                )
                    else:
                        draw.rectangle(
                                [rect_start_point, rect_end_point], outline=(255, 255, 255), width=6
                            )

                        draw.text(
                                    (rect_start_point[0], rect_start_point[1]-10), 
                                    text='unknown',
                                )
                        
                    # simi_list.append(simi)
                    # label_list.append(label)
                    # count += 1
                else:
                    draw.rectangle(
                                [rect_start_point, rect_end_point], outline=(255, 0, 0), width=6
                            )
                    
                    draw.text(
                                (rect_start_point[0], rect_start_point[1]-10), 
                                text='Spoof',
                            )
        imgbytes = cv2.imencode('.ppm', np.array(im_frame)[:, :, ::-1])[1].tobytes()
        # ---------------------------- THE GUI ----------------------------
        if not win_started:
            win_started = True
            layout = [
                [sg.Text('DeCloakFace', size=(30, 1))],
                [sg.Graph((W, H), (0,0), (W,H), key='-GRAPH-')],
                [sg.Text('FAS_threshold'),
                sg.Slider(range=(0, 20), orientation='h', resolution=1, default_value=16, size=(15, 15), key='FAS_threshold'),
                sg.Text('simi_threshold'),
                sg.Slider(range=(0, 20), orientation='h', resolution=1, default_value=10, size=(15, 15), key='simi_threshold')],
                [sg.Exit()]
            ]
            window = sg.Window('YOLO Webcam Demo', layout, default_element_size=(14, 1), text_justification='right', auto_size_text=False, finalize=True)
            image_elem = window['-GRAPH-']     # type: sg.Graph
        else:
            image_elem.erase()
            image_elem.draw_image(data=imgbytes, location=(0, H))

        event, values = window.read(timeout=0)
        if event is None or event == 'Exit':
            break
        
        
        fas_threshold = int(values['FAS_threshold']) / 20
        simi_threshold = int(values['simi_threshold']) / 20


    print("[INFO] cleaning up...")
    window.close()