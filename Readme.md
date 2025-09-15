# DanceOff

**A Just-Dance–style game using MediaPipe & OpenCV**

---

## Description

This project is a CS Fair entry built in Python. It uses **MediaPipe** for pose detection and **OpenCV** for video processing to create an interactive dance game similar to *Just Dance*. The player follows dance moves, and the system detects poses, compares them to trained models, and scores or gives feedback.

---

## Features

- Real-time pose detection using webcam input  
- Collection of pose data for training (poses for each dance move)  
- Training pipeline using scikit-learn to build pose classification models  
- Leaderboards / tracking of player scores  
- Overlay & visual feedback of current pose vs. target pose  
- Multiple dancers / multiple users supported  

---

## Getting Started

### Prerequisites

- Python 3.x  
- Libraries: `mediapipe`, `opencv-python`, `scikit-learn`, others (check `requirements.txt`)  

### Installation

```bash
git clone https://github.com/disisid17/DanceOff.git
cd DanceOff
pip install -r requirements.txt
