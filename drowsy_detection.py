from math import e
from re import M
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from sympy import plot


def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh
    # Define the indices for mouth landmarks
chosen_mouth_upper_idxs = [61, 62, 63, 64, 65, 66, 67]  # Upper lip
chosen_mouth_lower_idxs = [291, 308, 324, 318, 402, 317, 14]  # Lower lip

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def get_mar(landmarks, upper_idxs, lower_idxs, frame_width, frame_height):
    """
    Calculate Mouth Aspect Ratio (MAR) for yawning detection.

    Args:
        landmarks: (list) Detected landmarks list
        upper_idxs: (list) Index positions of the upper lip landmarks
        lower_idxs: (list) Index positions of the lower lip landmarks
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        mar: (float) Mouth aspect ratio
    """
    try:
        upper_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in upper_idxs]
        lower_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in lower_idxs]

        # Calculate vertical distances (upper lip to lower lip)
        vertical_distances = [distance(upper_lip_points[i], lower_lip_points[i]) for i in range(len(upper_idxs))]

        # Calculate horizontal distance (average distance between corners of the mouth)
        horizontal_distance = distance(upper_lip_points[0], upper_lip_points[-1])

        # Calculate MAR

        upper_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in upper_idxs]
        lower_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in lower_idxs]
        mar = (
            np.linalg.norm(np.array(upper_lip_points[3]) - np.array(lower_lip_points[3])) +
            np.linalg.norm(np.array(upper_lip_points[2]) - np.array(lower_lip_points[5])) +
            np.linalg.norm(np.array(upper_lip_points[4]) - np.array(lower_lip_points[1]))
        ) / (3 * np.linalg.norm(np.array(upper_lip_points[0]) - np.array(upper_lip_points[6])))

    except Exception as e:
        mar = 0.0
        upper_lip_points = lower_lip_points = None

    return mar, (upper_lip_points, lower_lip_points)


def calculate_avg_mar(landmarks, upper_idxs, lower_idxs, image_w, image_h):
    # Calculate Mouth aspect ratio

    upper_mar, upper_lm_coordinates = get_mar(landmarks, upper_idxs, image_w, image_h)
    lower_mar, lower_lm_coordinates = get_mar(landmarks, lower_idxs, image_w, image_h)
    Avg_MAR = (upper_mar + lower_mar) / 2.0

    return Avg_MAR, (upper_lm_coordinates, lower_lm_coordinates)

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio

    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)



def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    # Ensure frame is writable
    frame = frame.copy()

    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                try:
                    cv2.circle(frame, coord, 2, color, -1)
                except Exception as e:
                    print(f"Error drawing circle: {e}")

    return frame  # Remove redundant flipping if already done
def plot_lip_landmarks(frame, upper_lip_points, lower_lip_points, color):
    # Ensure frame is writable
    frame = frame.copy()

    for lm_coordinates in [upper_lip_points, lower_lip_points]:
        if lm_coordinates:
            for coord in lm_coordinates:
                try:
                    cv2.circle(frame, coord, 2, color, -1)
                except Exception as e:
                    print(f"Error drawing circle: {e}")

# def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
#     image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
#     return image
def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    # origin should be a tuple of two integers (x, y)
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class VideoFrameHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
            
        }
        self.mouth_idx={
            "upper": [61, 62, 63, 64, 65, 66, 67],
            "lower": [291, 308, 324, 318, 402, 317, 14]
        }
        

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "YAWNING_TIME": 0.0,
            "COLOR": self.GREEN,
            "COLOR_RED": self.RED,
            "play_alarm": False,
        }
        
        self.EAR_txt_pos = (10, 30)

    def process(self, frame: np.array, thresholds: dict):
      

        # To improve performance,
        # mark the frame as not writeable to pass by reference.
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Calculate EAR and MAR
            EAR, ear_coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
            MAR, mar_coordinates = get_mar(landmarks, self.mouth_idx["upper"], self.mouth_idx["lower"], frame_w, frame_h)

            # Plot EAR and MAR landmarks
            frame = plot_eye_landmarks(frame, ear_coordinates[0], ear_coordinates[1], self.state_tracker["COLOR"])
            # frames = plot_lip_landmarks(frames, mar_coordinates[0], mar_coordinates[1], self.state_tracker["COLOR"])
            # print("This is frames: "+frames)

            if EAR < thresholds["EAR_THRESH"] or MAR > thresholds["MAR_THRESH"]:
                
                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()
                
                # Increase DROWSY_TIME or YAWNING_TIME based on the condition met
                time_key = "DROWSY_TIME" if EAR < thresholds["EAR_THRESH"] else "YAWNING_TIME"
                self.state_tracker[time_key] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                # Trigger alarm if the time exceeds the WAIT_TIME threshold
                if self.state_tracker[time_key] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    alarm_text = "Drowsiness Detected" if EAR < thresholds["EAR_THRESH"] else "Yawning Detected"
                    plot_text(frame, alarm_text, ALM_txt_pos, self.state_tracker["COLOR"])

            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["YAWNING_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False
            
            EAR_txt = f"EAR: {round(EAR, 2)}"
            MAR_txt = f"MAR: {round(MAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            YAWNING_TIME_txt = f"YAWNING: {round(self.state_tracker['YAWNING_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, (10,25), self.state_tracker["COLOR"])
            plot_text(frame, MAR_txt, (10,50), self.state_tracker["COLOR"])
            plot_text(frame, YAWNING_TIME_txt, (10,350), self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, (10,380), self.state_tracker["COLOR"])

            # Flip the frame horizontally for a selfie-view display.
            # frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]

