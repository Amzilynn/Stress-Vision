# Video-Based Stress Analysis System

This project implements a **video-based stress analysis pipeline** using **OpenCV**, **MediaPipe**, and geometric computations.  
It estimates an **overall stress level over time** by analyzing multiple behavioral and facial cues extracted from a video stream.


---

##  Features
The system analyzes the following indicators:

-  **Blink frequency**
-  **Eyebrow movement**
-  **Facial emotion proxies**
-  **Lip movement**
-  **Hand movement intensity**
-  **Gaze direction**
-  **Face orientation**

All indicators are combined into a single **stress score**.

---

##  Stress Score Formula

The overall stress score is computed as a weighted sum:

```
Stress =
0.15 × Blink
+ 0.15 × Eyebrow
+ 0.15 × Emotions
+ 0.15 × Lips
+ 0.15 × Hand Movement
+ 0.15 × Gaze Direction
+ 0.10 × Face Orientation
```

---

##  Technologies Used

- **Python 3**
- **OpenCV** – video capture and processing
- **MediaPipe**
  - Face Detection
  - Holistic Model (face, hands, pose)
- **Matplotlib** – visualization
- **Math** – geometric computations

---

##  Project Structure

```
.
├── stress_analysis.py
├── stress_graph.png
└── README.md
```

---

##  How the System Works

1. Load a video file
2. Read frames using OpenCV
3. Detect face and body landmarks using MediaPipe
4. Extract behavioral features per frame
5. Compute individual stress indicators
6. Calculate the final stress score
7. Plot stress variation over time

---

##  Installation

Install the required Python packages:

```bash
pip install opencv-python mediapipe matplotlib
```

---

##  Usage

1. Update the video path in the script:

```python
video_path = ""
```

2. Run the script:

```bash
python stress_analysis.py
```

---

## Output

- Stress values computed per frame
- Line plot showing stress evolution over time
- Graph saved as:

```
stress_graph.png
```

---

##  Future Improvements

- Deep learning–based emotion recognition
- Improved blink detection
- Stress normalization and scaling
- Real-time webcam support
- Temporal modeling (LSTM, smoothing filters)

---

