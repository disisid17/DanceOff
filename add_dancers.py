import cv2
import mediapipe as mp
import imutils
import argparse
import copy

import numpy as np

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_pose = mp.solutions.pose

DEFAULT_IMAGE_WIDTH=1200
X_TRANSLATION_PIXELS=200
Z_TRANSLATION_PIXELS=100

"""
Usage:

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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("--image-width", type=int, required=False, default=1200,
                    help="Image width")
    args = vars(ap.parse_args())

    DEFAULT_IMAGE_WIDTH = args['image_width']

    ease_out = np.linspace(0.0, X_TRANSLATION_PIXELS, num=10)
    ease_out_index = 0
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
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

            # add dancers in this section
            if results:
                if results.pose_landmarks:

                    x_tran = X_TRANSLATION_PIXELS
                    if ease_out_index < len(ease_out):
                        x_tran = ease_out[ease_out_index]
                        ease_out_index += 1

                    add_dancer(results.pose_landmarks, x_tran)
                    add_dancer(results.pose_landmarks, -x_tran)
                    add_dancer(results.pose_landmarks, 2*-x_tran)
                    # add_dancer(results.pose_landmarks, -X_TRANSLATION_PIXELS, -Z_TRANSLATION_PIXELS)

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('Pose', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
