import cv2
import numpy as np
import mediapipe as mp
import pygame  # Import pygame for playing sound

# Initialize MediaPipe Face Mesh
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Define the indices for left and right eye landmarks
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]

# Define the indices for mouth landmarks
chosen_mouth_upper_idxs = [61, 62, 63, 64, 65, 66, 67]  # Upper lip
chosen_mouth_lower_idxs = [291, 308, 324, 318, 402, 317, 14]  # Lower lip

# Initialize pygame for audio playback
pygame.init()
pygame.mixer.init()
wake_up_sound = pygame.mixer.Sound('audio/wake_up.wav')  # Load wake up sound
be_alert_sound = pygame.mixer.Sound('clap.wav')  # Load be alert sound

# Thresholds and counters for detection
EAR_THRESHOLD = 0.10  # Threshold for EAR, indicating drowsiness
MAR_THRESHOLD = 0.8  # Threshold for MAR, indicating yawning
FRAMES_TO_ALARM = 48  # Frames below EAR threshold before alarm
frame_counter = 0  # Counter for consecutive frames below EAR threshold
yawn_counter = 0  # Counter for frames with yawning

# Function to calculate the Eye Aspect Ratio (EAR)
def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)
        P2_P6 = np.linalg.norm(np.array(coords_points[1]) - np.array(coords_points[5]))
        P3_P5 = np.linalg.norm(np.array(coords_points[2]) - np.array(coords_points[4]))
        P1_P4 = np.linalg.norm(np.array(coords_points[0]) - np.array(coords_points[3]))
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
    except Exception as e:
        ear = 0.0
    return ear

# Function to calculate the Mouth Aspect Ratio (MAR)
def get_mar(landmarks, upper_idxs, lower_idxs, frame_width, frame_height):
    upper_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in upper_idxs]
    lower_lip_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in lower_idxs]
    mar = (
        np.linalg.norm(np.array(upper_lip_points[3]) - np.array(lower_lip_points[3])) +
        np.linalg.norm(np.array(upper_lip_points[2]) - np.array(lower_lip_points[5])) +
        np.linalg.norm(np.array(upper_lip_points[4]) - np.array(lower_lip_points[1]))
    ) / (3 * np.linalg.norm(np.array(upper_lip_points[0]) - np.array(upper_lip_points[6])))
    return mar

# Initialize Video Capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

with mp_facemesh.FaceMesh(refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                imgH, imgW, _ = frame.shape
                landmarks = face_landmarks.landmark

                # Calculate EAR for both eyes
                EAR = (get_ear(landmarks, chosen_left_eye_idxs, imgW, imgH) + 
                       get_ear(landmarks, chosen_right_eye_idxs, imgW, imgH)) / 2

                # Calculate MAR
                MAR = get_mar(landmarks, chosen_mouth_upper_idxs, chosen_mouth_lower_idxs, imgW, imgH)

                # Display EAR and MAR
                cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Drowsiness detection
                if EAR < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= FRAMES_TO_ALARM:
                        pygame.mixer.Sound.play(wake_up_sound)
                        cv2.putText(frame, "WAKE UP!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    frame_counter = 0

                # Yawning detection
                if MAR > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= 5:  # Adjust as needed
                        pygame.mixer.Sound.play(be_alert_sound)
                        cv2.putText(frame, "BE ALERT, BE ACTIVE!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    yawn_counter = 0

        cv2.imshow('Drowsiness and Yawning Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
