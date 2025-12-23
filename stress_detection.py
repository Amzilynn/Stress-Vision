import cv2 
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_stress(blink, eyebrow, emotions, lips, hand_movement, gaze_direction, face_orientation):
    """Calculate overall stress from individual metrics"""
    final_stress = (
        0.15 * blink +
        0.15 * eyebrow +
        0.15 * emotions +
        0.15 * lips +
        0.15 * hand_movement +
        0.15 * gaze_direction +
        0.10 * face_orientation
    )
    return final_stress

def calculate_hand_movement(hand_landmarks):
    """Calculate total hand movement based on landmark distances"""
    if hand_landmarks is None:
        return 0
    
    total_distance = 0
    for i in range(1, len(hand_landmarks.landmark)):
        x1, y1, z1 = hand_landmarks.landmark[i-1].x, hand_landmarks.landmark[i-1].y, hand_landmarks.landmark[i-1].z
        x2, y2, z2 = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        total_distance += distance

    return min(total_distance * 10, 100)  # Normalize to 0-100

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_angle(point1, point2, point3, point4):
    """Calculate angle between two vectors"""
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4
    
    vector1 = (x2 - x1, y2 - y1)
    vector2 = (x4 - x3, y4 - y3)
    
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    if magnitude1 != 0 and magnitude2 != 0:
        angle_rad = math.acos(max(-1, min(1, dot_product / (magnitude1 * magnitude2))))
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    else:
        return 0

def analyze_blink(face_landmarks, frame_height):
    """Analyze eye aspect ratio to detect blinks"""
    if face_landmarks is None:
        return 0
    
    # Left eye landmarks (approximate)
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    left_eye_left = face_landmarks.landmark[33]
    left_eye_right = face_landmarks.landmark[133]
    
    # Calculate eye aspect ratio
    vertical_dist = abs(left_eye_top.y - left_eye_bottom.y) * frame_height
    horizontal_dist = abs(left_eye_left.x - left_eye_right.x) * frame_height
    
    if horizontal_dist > 0:
        ear = vertical_dist / horizontal_dist
        # Lower EAR suggests closed eyes (blink or tiredness)
        blink_score = max(0, (0.2 - ear) * 500)  # Normalize
        return min(blink_score, 100)
    
    return 0

def analyze_eyebrow(face_landmarks):
    """Analyze eyebrow position and movement"""
    if face_landmarks is None:
        return 0
    
    # Eyebrow landmarks
    left_eyebrow_inner = face_landmarks.landmark[70]
    left_eyebrow_outer = face_landmarks.landmark[46]
    right_eyebrow_inner = face_landmarks.landmark[300]
    right_eyebrow_outer = face_landmarks.landmark[276]
    
    # Eye landmarks for reference
    left_eye = face_landmarks.landmark[159]
    right_eye = face_landmarks.landmark[386]
    
    # Calculate eyebrow height relative to eyes
    left_height = abs(left_eyebrow_inner.y - left_eye.y)
    right_height = abs(right_eyebrow_inner.y - right_eye.y)
    avg_height = (left_height + right_height) / 2
    
    # Higher eyebrows can indicate stress/surprise
    eyebrow_score = max(0, (avg_height - 0.03) * 1000)
    return min(eyebrow_score, 100)

def analyze_emotions(face_landmarks):
    """Analyze facial expressions for emotion indicators"""
    if face_landmarks is None:
        return 0
    
    # Mouth corners
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]
    mouth_center = face_landmarks.landmark[13]
    
    # Calculate mouth curvature (smile/frown)
    left_curve = left_mouth.y - mouth_center.y
    right_curve = right_mouth.y - mouth_center.y
    avg_curve = (left_curve + right_curve) / 2
    
    # Negative curve (frown) indicates stress
    emotion_score = max(0, avg_curve * 500)
    return min(emotion_score, 100)

def analyze_lips(face_landmarks):
    """Analyze lip compression and tension"""
    if face_landmarks is None:
        return 0
    
    # Upper and lower lip landmarks
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    
    # Calculate lip distance
    lip_distance = abs(upper_lip.y - lower_lip.y)
    
    # Compressed lips (small distance) can indicate stress
    lip_score = max(0, (0.02 - lip_distance) * 2000)
    return min(lip_score, 100)

def analyze_hand_movement(hand_landmarks):
    """Wrapper for hand movement calculation"""
    return calculate_hand_movement(hand_landmarks)

def analyze_gaze_direction(face_landmarks):
    """Analyze gaze direction and stability"""
    if face_landmarks is None:
        return 0
    
    # Iris landmarks (approximate)
    left_iris = face_landmarks.landmark[468]
    left_eye_center = face_landmarks.landmark[33]
    
    # Calculate horizontal deviation
    gaze_deviation = abs(left_iris.x - left_eye_center.x)
    
    # More deviation can indicate distraction/stress
    gaze_score = min(gaze_deviation * 500, 100)
    return gaze_score

def analyze_face_orientation(pose_landmarks):
    """Analyze head pose and orientation"""
    if pose_landmarks is None:
        return 0
    
    # Use nose and ear landmarks to estimate head rotation
    nose = pose_landmarks.landmark[0]
    left_ear = pose_landmarks.landmark[7]
    right_ear = pose_landmarks.landmark[8]
    
    # Calculate asymmetry (head tilt)
    left_dist = abs(nose.x - left_ear.x)
    right_dist = abs(nose.x - right_ear.x)
    asymmetry = abs(left_dist - right_dist)
    
    # More asymmetry can indicate stress/discomfort
    orientation_score = min(asymmetry * 200, 100)
    return orientation_score

def analyze_video(video_path):
    """Main function to analyze video and extract stress metrics"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    stress_data = {
        'blink': [], 
        'eyebrow': [], 
        'emotions': [], 
        'lips': [],
        'hand_movement': [], 
        'gaze_direction': [], 
        'face_orientation': [], 
        'overall': []
    }
    
    frame_count = 0
    print("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Progress update every 30 frames
            print(f"Processing frame {frame_count}...")
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height = frame.shape[0]
        
        # Process frame with holistic model
        results = holistic.process(rgb_frame)
        
        # Extract metrics
        blink = analyze_blink(results.face_landmarks, frame_height)
        eyebrow = analyze_eyebrow(results.face_landmarks)
        emotions = analyze_emotions(results.face_landmarks)
        lips = analyze_lips(results.face_landmarks)
        hand_movement = analyze_hand_movement(results.right_hand_landmarks)
        gaze_direction = analyze_gaze_direction(results.face_landmarks)
        face_orientation = analyze_face_orientation(results.pose_landmarks)
        
        # Store data
        stress_data['blink'].append(blink)
        stress_data['eyebrow'].append(eyebrow)
        stress_data['emotions'].append(emotions)
        stress_data['lips'].append(lips)
        stress_data['hand_movement'].append(hand_movement)
        stress_data['gaze_direction'].append(gaze_direction)
        stress_data['face_orientation'].append(face_orientation)
        
        # Calculate overall stress
        overall_stress = calculate_stress(
            blink, eyebrow, emotions, lips, 
            hand_movement, gaze_direction, face_orientation
        )
        stress_data['overall'].append(overall_stress)
    
    cap.release()
    holistic.close()
    
    print(f"Video analysis complete! Processed {frame_count} frames.")
    return stress_data

def plot_graph(stress_data):
    """Generate and save stress visualization"""
    if not stress_data or len(stress_data['overall']) == 0:
        print("No data to plot!")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot overall stress
    axes[0].plot(stress_data['overall'], linewidth=2, color='red')
    axes[0].set_title('Overall Stress Levels Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Stress Level (0-100)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])
    
    # Plot individual metrics
    axes[1].plot(stress_data['blink'], label='Blink', alpha=0.7)
    axes[1].plot(stress_data['eyebrow'], label='Eyebrow', alpha=0.7)
    axes[1].plot(stress_data['emotions'], label='Emotions', alpha=0.7)
    axes[1].plot(stress_data['lips'], label='Lips', alpha=0.7)
    axes[1].plot(stress_data['hand_movement'], label='Hand Movement', alpha=0.7)
    axes[1].plot(stress_data['gaze_direction'], label='Gaze', alpha=0.7)
    axes[1].plot(stress_data['face_orientation'], label='Face Orientation', alpha=0.7)
    
    axes[1].set_title('Individual Stress Metrics', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('Metric Value (0-100)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('stress_graph.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'stress_graph.png'")
    plt.show()
    
    # Print statistics
    print("\n=== Stress Analysis Statistics ===")
    print(f"Average Overall Stress: {np.mean(stress_data['overall']):.2f}")
    print(f"Maximum Stress: {np.max(stress_data['overall']):.2f}")
    print(f"Minimum Stress: {np.min(stress_data['overall']):.2f}")
    print(f"Standard Deviation: {np.std(stress_data['overall']):.2f}")

if __name__ == "__main__":
    # Update this path to your video file
    video_path = ""
    
    # Analyze video
    stress_data = analyze_video(video_path)
    
    # Generate graph
    if stress_data:
        plot_graph(stress_data)
