# Import necessary libraries
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase

import os, pickle
import numpy as np
from PIL import Image

# Import necessary libraries for DTW
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from collections import deque

def get_starting_pose(exercise_name):
    pose_dict = {}
    current_path = os.path.dirname(__file__)
    starting_poses_dir = os.path.join(current_path, "exercise_starting_poses")
    starting_poses = [f for f in os.listdir(starting_poses_dir)]

    for pose in starting_poses:
        pose_name = os.path.basename(pose).split('.')[0]
        pose_dict[pose_name] = os.path.join(starting_poses_dir, pose)
    pose_path = pose_dict[exercise_name]

    return pose_path

def run_keypoints_generator(pose_model):
    keypoints_dict = {}

    current_path = os.path.dirname(__file__)
    reference_exercises_dir = os.path.join(current_path, "exercise_shortened")
    exercise_videos = [f for f in os.listdir(reference_exercises_dir) if f.endswith(".mp4")]
    keypoints_folder = os.path.join("exercise_keypoints")

    for video_file in exercise_videos:
        keypoints_file = os.path.join(keypoints_folder, str(video_file.split(".")[0]+".pkl"))

        if not os.path.exists(keypoints_file):
            video_path = os.path.join(reference_exercises_dir, video_file)
            keypoints = process_video(video_path, pose_model)
            save_keypoints(keypoints, keypoints_file)
        else:
            keypoints = load_keypoints(keypoints_file)

        exercise_name = os.path.basename(video_file).split('.')[0]  # Assuming the video name is the exercise name
        keypoints_dict[exercise_name] = keypoints

    return keypoints_dict

def process_video(video_path, pose_model):
    cap = cv2.VideoCapture(video_path)
    pose_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # RGB conversion
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Pose estimation
        results = pose_model.process(frame)
        # Check if any pose is detected
        if results.pose_landmarks:
            keypoints = extract_keypoints(results)
            pose_landmarks.append(keypoints)

    cap.release()
    return np.array(pose_landmarks)

def save_keypoints(keypoints, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(keypoints, f)

def load_keypoints(file_path):
    with open(file_path, 'rb') as f:
        keypoints = pickle.load(f)
    return keypoints

def extract_keypoints(results):
    keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    return keypoints

def normalize_keypoints(keypoints):
    min_val = np.min(keypoints)
    max_val = np.max(keypoints)

    if max_val - min_val == 0:
        return keypoints - min_val
    else:
        return (keypoints - min_val) / (max_val - min_val)


# Keypoints comparison using DTW
def compare_keypoints(user_keypoints_sequence, reference_keypoints, window_size):
    min_distance = float('inf')

    # 'slide the window' along the reference keypoints
    for i in range(len(reference_keypoints) - window_size + 1):
        # windowed_reference_keypoints = normalize_keypoints(reference_keypoints[i:i + window_size])
        windowed_reference_keypoints = reference_keypoints[i:i + window_size]

        if np.any(np.isnan(user_keypoints_sequence)) or np.any(np.isinf(user_keypoints_sequence)):
            print("WARNING: USER KEYPOINTS SEQUENCE ============================================")
            print(f"NaN? {np.any(np.isnan(user_keypoints_sequence))}\n Inf? {np.any(np.isinf(user_keypoints_sequence))}")
            user_keypoints_sequence = np.nan_to_num(user_keypoints_sequence)
        if np.any(np.isnan(windowed_reference_keypoints)) or np.any(np.isinf(windowed_reference_keypoints)):
            print("WARNING: WINDOWED REFERENCE KEYPOINTS ============================================")
            print(f"NaN? {np.any(np.isnan(windowed_reference_keypoints))}\n Inf? {np.any(np.isinf(windowed_reference_keypoints))}")
            windowed_reference_keypoints = np.nan_to_num(windowed_reference_keypoints)

        dist, path = fastdtw(user_keypoints_sequence, windowed_reference_keypoints, dist=euclidean)
        if dist < min_distance:
            min_distance = dist

    return (min_distance / window_size)

# Generating feedback
def generate_feedback(min_distance, distance_threshold, selected_exercise):
    # If the minimum distance is below a certain threshold, tell the user their form is good
    if min_distance < (min_distance + (distance_threshold/2)):
        feedback_message = f"Your form is good for {selected_exercise}!"
    else:
        feedback_message = f"Your form could be improved."
    return feedback_message

# Define a video transformer class to process the video
# Extend the VideoTransformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self, keypoints_dict, selected_exercise, pose_model):
        self.pose = pose_model
        self.selected_exercise = selected_exercise
        self.reference_keypoints = keypoints_dict[selected_exercise]
        self.repetition_count = 0
        self.frame_count = 0

        self.frames_per_rep = {"Basic-Burpees":56,"Crunches":35,"Jumping-Jacks":28,
                                "Leg-Raises":37,"Push-Ups":88,"Sit-Ups":72,"Slow-Climbers":59,"Squats":45}
        self.dtw_thresholds = {"Basic-Burpees": 1.5, "Crunches": 1, "Jumping-Jacks": 1.05,
                               "Leg-Raises": 1.15, "Push-Ups": 1.18, "Sit-Ups": 1, "Slow-Climbers": 0.57, "Squats": 1.3}

        self.window_size = len(self.reference_keypoints) # Number of frames per rep based on the reference video
        self.feedback_frequency = round(self.window_size / 3)  # Give feedback every x frames
        self.distance_threshold = self.dtw_thresholds[selected_exercise]

        self.repetition_threshold = None
        self.user_initial_keypoints = []
        self.end_reference_keypoints = np.mean(self.reference_keypoints[-5:], axis=0) # Average of last 5 keypoints in the reference vid

        self.end_of_rep = False
        self.wait_time = self.window_size * 0.8
        self.frame_of_last_rep = 0

        self.min_distance = 0
        self.feedback_message = "No feedback."
        self.user_keypoints_sequence = deque(maxlen=self.window_size)

        self.dtw_distances = deque(maxlen=self.feedback_frequency)
        self.dtw_average = 0

        self.calibration = True

    def transform(self, frame):
        # Convert the frame to a numpy array
        conv_frame = frame.to_ndarray(format="bgr24")
        # Convert the frame color to RGB
        conv_frame = cv2.cvtColor(conv_frame, cv2.COLOR_BGR2RGB)
        self.frame_count += 1

        # Perform pose estimation
        results = self.pose.process(conv_frame)

        # Extract the keypoints
        if results.pose_landmarks:
            user_keypoints = extract_keypoints(results)

            # Set dynamic repetition threshold based on first 5 frames of user keypoints
            if self.repetition_threshold is None:
                self.user_initial_keypoints.append(user_keypoints)
                if len(self.user_initial_keypoints) >= self.window_size:  # calculate threshold after first [window_size] frames
                    distances = [np.linalg.norm(uk - self.end_reference_keypoints) for uk in self.user_initial_keypoints]
                    self.repetition_threshold = np.mean(distances)

            self.user_keypoints_sequence.append(user_keypoints)

            # If the sequence is long enough, compare it with the reference poses
            if len(self.user_keypoints_sequence) >= self.window_size:
                self.calibration = False
                user_keypoints_sequence = np.array(self.user_keypoints_sequence)

                user_keypoints_sequence = normalize_keypoints(user_keypoints_sequence)
                reference_keypoints = normalize_keypoints(self.reference_keypoints)

                # Get the minimum distance by comparing the user's pose with the reference poses
                self.min_distance = compare_keypoints(user_keypoints_sequence, reference_keypoints, self.window_size)

                self.dtw_distances.append(self.min_distance)
                self.dtw_average = np.max(self.dtw_distances)

                if self.frame_count % self.feedback_frequency == 0:  # Only give feedback every 30 frames
                    self.feedback_message = generate_feedback(self.min_distance, self.distance_threshold, self.selected_exercise)

                # Counting repetitions
                # dist_to_end = np.linalg.norm(user_keypoints - self.end_reference_keypoints) # distance of current keypoints from the end_keypoints
                dist_to_end = np.mean([np.linalg.norm(uk - self.end_reference_keypoints) for uk in self.user_initial_keypoints])

                if self.repetition_threshold != None:
                    # if dist_to_end <= self.repetition_threshold:
                    if self.min_distance <= (self.repetition_threshold+(self.distance_threshold/5)):
                        self.end_of_rep = True
                    elif dist_to_end > self.repetition_threshold:
                        self.end_of_rep = False

                if (self.end_of_rep and (self.frame_count > (self.frame_of_last_rep + self.wait_time))):
                    self.repetition_count += 1
                    self.frame_of_last_rep = self.frame_count
                    self.end_of_rep = False

        # Draw the pose estimation annotations on the frame
            mp.solutions.drawing_utils.draw_landmarks(conv_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        if self.calibration:
            conv_frame = outlined_text(conv_frame, f"Calibrating..", (10, 30))

        else:
            # Draw the repetition count on the frame
            conv_frame = outlined_text(conv_frame, f"Repetitions: {self.repetition_count}", (10, 30))

            # Display the DTW distances and the feedback message
            conv_frame = outlined_text(conv_frame, f"DTW Distance: {round(self.min_distance, 4)}", (10, 60))
            conv_frame = outlined_text(conv_frame, self.feedback_message, (10, 90))

            # Convert the frame color back to BGR for display
            conv_frame = cv2.cvtColor(conv_frame, cv2.COLOR_RGB2BGR)

        return conv_frame

def outlined_text(img, text, pos, font_scale=1, color=(205,205,205), outline_color=(0,128,128), thickness=(2), outline_thickness=(3)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return img

def highlight_keypoint(frame, point, radius=5, color=(0, 255, 0), thickness=2):
    # Assume point is a tuple of (x, y)
    cv2.circle(frame, point, radius, color, thickness)
    return frame

def find_closest_frame(user_keypoints, reference_keypoints):
    min_diff = float('inf')
    closest_frame = None
    for frame in reference_keypoints:
        diff = np.sum(np.abs(user_keypoints - frame))
        if diff < min_diff:
            min_diff = diff
            closest_frame = frame
    return closest_frame

# Create a Streamlit application
def main():
    st.header("Exercise Coach")

    # Initialize the MediaPipe Pose object
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Generate or Load the generated keypoints from the reference exercise videos
    keypoints_dict = run_keypoints_generator(mp_pose)

    # Add a selectbox for the user to choose the exercise
    exercise = st.selectbox("Choose an exercise", list(keypoints_dict.keys()))

    # Define the configuration of the WebRTC connection
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    # Start the webcam feed and display it in the app
    webrtc_streamer(key="camera_feed",rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=lambda: VideoTransformer(keypoints_dict, exercise, mp_pose),
                    media_stream_constraints={"video": True, "audio": False})

    # Tell the user to follow the starting pose for calibration
    st.write("Please maintain the starting pose during calibration:")
    starting_pose = Image.open(get_starting_pose(exercise))
    st.image(starting_pose, f"Starting Pose for {exercise}")

if __name__ == "__main__":
    main()