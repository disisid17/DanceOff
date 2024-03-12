import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import imutils
import argparse
import copy

DEFAULT_IMAGE_WIDTH = 1200
X_TRANSLATION_PIXELS = 200
Z_TRANSLATION_PIXELS = 100

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose

"""
Usage:

python 03_pose_predictions.py --model-name best_ymca_pose_model

python 03_pose_predictions.py 
"""


def add_dancer(landmark_values, x_translation_pixels, z_translation_pixels=None):
    landmarks_copy = copy.deepcopy(landmark_values)
    if landmarks_copy:
        for i, lm in enumerate(landmarks_copy.landmark):
            lm.x = lm.x + x_translation_pixels / DEFAULT_IMAGE_WIDTH
            if z_translation_pixels is not None:
                lm.z = lm.z + z_translation_pixels / DEFAULT_IMAGE_WIDTH

        mp_drawing.draw_landmarks(image, landmarks_copy, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )


def retpos():
##    ap = argparse.ArgumentParser()
##
##    ap.add_argument("--model-name", type=str, required=False, default='best_ymca_pose_model',
##                    help="name of the saved pickled model [no suffix]")
##    ap.add_argument("--suppress-landmarks", action='store_true',
##                    help="[Optional: False] if present do not show landmarks on yourself ")
##    ap.add_argument("--image-width", type=int, required=False, default=1200,
##                    help="Image width")
##    ap.add_argument("--add-dancers", action='store_true',
##                    help="[Optional: False] add the rest of the virtual village people")
##    ap.add_argument("--add-counters", action='store_true',
##                    help="[Optional: False] if present should the pose counts ")
##
##    args = vars(ap.parse_args())
    #DEFAULT_IMAGE_WIDTH = args['image_width']
    add_dancers = False

    model_name = 'best_ymca_pose_model'
    suppress_landmarks = False
    add_counters = False

    with open(f'{model_name}.pkl', 'rb') as f:
        model = joblib.load(f)

    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    _,fram = cap.read()
    DEFAULT_IMAGE_WIDTH = fram.shape[0]
    y_counter = 0
    m_counter = 0
    c_counter = 0
    a_counter = 0
    last_detected_pose = None
    number_of_new_pose_detections = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        if cap.isOpened():
            ret, frame = cap.read()

            frame = imutils.resize(frame, width=DEFAULT_IMAGE_WIDTH)

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

##            if add_dancers:
##                if results:
##                    if results.pose_landmarks:
##                        add_dancer(results.pose_landmarks, X_TRANSLATION_PIXELS)
##                        add_dancer(results.pose_landmarks, -X_TRANSLATION_PIXELS)
##                        add_dancer(results.pose_landmarks, 2 * -X_TRANSLATION_PIXELS)

            # 4. Pose Detections
##            if not suppress_landmarks:
##                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
##                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
##                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
##                                          )
            # Export coordinates
            try:
                # Extract Pose landmarks
                landmarks = results.pose_landmarks.landmark
                arm_landmarks = []
                pose_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.LEFT_ELBOW.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.LEFT_WRIST.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                pose_index = mp_pose.PoseLandmark.RIGHT_WRIST.value
                arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y, landmarks[pose_index].z]

                row = np.around(arm_landmarks, decimals=9).tolist()

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                #print(body_language_class)
                return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "BGR"),body_language_prob
           
               

##                if add_counters:
##                    if last_detected_pose != body_language_class and np.around(np.max(body_language_prob),
##                                                                               decimals=3) > 0.700:
##                        print(last_detected_pose, body_language_class)
##                        last_detected_pose = body_language_class
##                        if body_language_class == 'Y':
##                            y_counter += 1
##                        elif body_language_class == 'M':
##                            m_counter += 1
##                        elif body_language_class == 'C':
##                            c_counter += 1
##                        elif body_language_class == 'A':
##                            a_counter += 1

                # Get status box
##                status_width = 250
##                if add_counters:
##                    status_width = 500
##                cv2.rectangle(image, (0, 0), (status_width, 60), (245, 117, 16), -1)

                # Display Class
##                cv2.putText(image, 'CLASS'
##                            , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                #cv2.putText(image, body_language_class.split(' ')[0]
                           # , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Probability
##                cv2.putText(image, 'PROB'
##                            , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
##                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
##                            , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
               #print(str(round(body_language_prob[np.argmax(body_language_prob)], 2)))
##                if add_counters:
##                    # Display Y count
##                    count_x_start = 210
##                    count_x_offset = 50
##                    value_x_offset = 5
##
##                    count_x = count_x_start
##                    cv2.putText(image, 'Y'
##                                , (count_x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
##                    cv2.putText(image, str(y_counter)
##                                , (count_x - value_x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
##                                cv2.LINE_AA)
##
##                    count_x = count_x + count_x_offset
##                    # Display M count
##                    cv2.putText(image, 'M'
##                                , (count_x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
##                    cv2.putText(image, str(m_counter)
##                                , (count_x - value_x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
##                                cv2.LINE_AA)
##
##                    count_x = count_x + count_x_offset
##                    # Display C count
##                    cv2.putText(image, 'C'
##                                , (count_x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
##                    cv2.putText(image, str(c_counter)
##                                , (count_x - value_x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
##                                cv2.LINE_AA)
##
##                    count_x = count_x + count_x_offset
##                    # Display A count
##                    cv2.putText(image, 'A'
##                                , (count_x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
##                    cv2.putText(image, str(a_counter)
##                                , (count_x - value_x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
##                                cv2.LINE_AA)
##


            except Exception as exc:
                #print(f"{exc}")
                return "None"
                pass

##            cv2.imshow('Pose Prediction', image)
##
##            if cv2.waitKey(10) & 0xFF == ord('q'):
##                break

##    cap.release()
##    cv2.destroyAllWindows()
