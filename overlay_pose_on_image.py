import mediapipe as mp  # Import mediapipe
import cv2  # Import opencv
import argparse

"""
Utility script to load an image and run it through the mediapipe models to draw any pose information

Usage:

python overlay_pose_on_image.py --image-path ./data/Y.png

python overlay_pose_on_image.py --image-path ./data/Y.png --output-path ./data/Y-pose.png

"""

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-path", required=True, help="path to image to load")
    ap.add_argument("--output-path", required=False, help="path/filename of image to write with pose")

    args = vars(ap.parse_args())

    image_path = args['image_path']
    output_image_path = args['output_path']

    # save an image  of the pose to so we can overlay points
    image = cv2.imread(image_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Pose Detection', image)
        if output_image_path:
            cv2.imwrite(output_image_path, image)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
