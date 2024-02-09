# yawning_detection.py

import numpy as np
import mediapipe as mp

# Mediapipe drawing utils for normalization of coordinates
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Define the indices for mouth landmarks
chosen_mouth_upper_idxs = [61, 62, 63, 64, 65, 66, 67]  # Upper lip
chosen_mouth_lower_idxs = [291, 308, 324, 318, 402, 317, 14]  # Lower lip

def get_mar(landmarks, upper_idxs, lower_idxs, frame_width, frame_height):
    """ Calculate the Mouth Aspect Ratio (MAR) for yawning detection """
    upper_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in upper_idxs]
    lower_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in lower_idxs]

    mar = np.linalg.norm(np.array(upper_lip_points[3]) - np.array(lower_lip_points[3])) + \
          np.linalg.norm(np.array(upper_lip_points[2]) - np.array(lower_lip_points[5])) + \
          np.linalg.norm(np.array(upper_lip_points[4]) - np.array(lower_lip_points[6]))
    mar /= np.linalg.norm(np.array(upper_lip_points[0]) - np.array(lower_lip_points[6]))

    return mar

# Define a threshold for yawning detection
MAR_THRESHOLD = 0.8
