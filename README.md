#  Real-Time Shoplifting Detection using Pose Estimation and Machine Learning

This project is a part of my internship, aimed at building a real-time suspicious behavior detection system using surveillance footage. The focus is on detecting **shoplifting behavior** by combining **YOLOv8-based pose estimation** with **XGBoost classification**. The system tracks people, analyzes their movement and body keypoints, and flags any potentially suspicious activity in live or recorded video streams.

---

##  Project Structure

```bash
HB/
├── dataset_path/
│   ├── Normal/               # Cropped person images (normal behavior)
│   ├── Suspicious/           # Cropped person images (suspicious behavior)
│   └── dataset.csv           # Final labeled keypoint dataset
├── images/                   # Full video frames (for reference)
├── images1/                  # Temporary cropped people (emptied later)
├── vid.mp4                   # Input video file
├── output_vid.mp4            # Final annotated output video
├── output_blurred.mp4        # Optional blurred video from GUI
├── nkeypoint.csv             # Raw keypoints from all detections
├── trained_model.json        # Trained XGBoost model
├── tksoft.py                 # GUI tool to blur selected people in a video
├── normal.py                 # Extracts keypoints from normal-only video
├── Suspicious.py             # Extracts keypoints from suspicious-only video
├── ImageShuffle.py           # Moves images into Normal/Suspicious folders
├── dataset.py                # Labels keypoints as Normal/Suspicious
├── model.py                  # Trains XGBoost model on keypoint data
└── main.py                   # Final pipeline for real-time shoplifting detection
```


---

##  How It Works

### 1. **Blurring (tksoft.py)**  
Interactive GUI lets you blur selected people by tracking IDs and create two filtered videos:  
- `normal_video.mp4`: shows only normal activity  
- `sus_video.mp4`: shows only suspicious people

### 2. **Keypoint Extraction (normal.py & Suspicious.py)**  
- Uses YOLOv8-pose to extract 17 keypoints for each person  
- Saves cropped person images and body pose coordinates

### 3. **Dataset Generation (ImageShuffle.py & dataset.py)**  
- Moves images to labeled folders (`Normal/`, `Suspicious/`)  
- Creates a labeled dataset (`dataset.csv`) for model training

### 4. **Model Training (model.py)**  
- Trains an XGBoost classifier to distinguish between normal and suspicious behavior using body keypoints  
- Outputs `trained_model.json`

### 5. **Real-Time Prediction (main.py)**  
- Processes `vid.mp4`  
- Detects people with YOLOv8-pose  
- Classifies behavior using the trained XGBoost model  
- Annotates suspicious people in red, normal in green → saves to `output_vid.mp4`

---

## Example Output

-  Bounding boxes labeled **"Suspicious"**  or **"Normal"**
-  Live video annotated and saved as `output_vid.mp4`
-  Console logs for predictions and detections per frame

---

## 🛠 Dependencies

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV (`opencv-python`)
- pandas
- xgboost
- numpy
- cvzone
- tkinter
- Pillow

Install dependencies with:
```bash
pip install -r requirements.txt
```

##  Outputs Generated

| File                 | Description                                           |
|----------------------|-------------------------------------------------------|
| `output_blurred.mp4` | Blurred video showing only normal or suspicious activity (created using GUI) |
| `dataset.csv`        | Final labeled pose keypoint dataset with image names and class labels |
| `trained_model.json` | Trained XGBoost classification model (used for predictions) |
| `output_vid.mp4`     | Final annotated video showing real-time predictions (Normal/Suspicious) |

---

##  Model Details

- **Pose Model**: `yolo11n-pose.pt`  
- **Classifier**: `XGBoost` (`binary:logistic`)  
- **Input Features**: 34 pose keypoints → `x0–x16`, `y0–y16`  
- **Output Classes**:
  - `0` = **Suspicious**
  - `1` = **Normal**

## Use Case
Surveillance systems in malls, retail stores, and parking lots

Helps operators focus attention on real-time suspicious behavior

Can be expanded to detect loitering, abnormal postures, etc.

