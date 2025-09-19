# DanceOff

A real-time interactive dance game where users follow dance moves and get scored based on pose accuracy.

---

## Table of Contents

1. [Motivation](#motivation)  
2. [Features](#features)  
3. [Demo](#demo)  
4. [Architecture & Design](#architecture--design)  
5. [Getting Started](#getting-started)  
   - Prerequisites  
   - Installation  
   - Usage  
6. [Code Structure](#code-structure)  
7. [Results & Limitations](#results--limitations)  
8. [Future Improvements](#future-improvements)  
9. [License](#license)

---

## Motivation

I built **DanceOff** for the 2024 CS Fair, aiming to merge computer vision with interactive entertainment. It combines pose estimation with visual feedback so users can dance and learn in real time.

---

## Features

- Pose detection via MediaPipe  
- Real-time video capture & processing using OpenCV  
- Training data collection for new dance moves  
- Pose matching / classification (scikit-learn or similar)  
- Live feedback overlay (showing target vs actual pose)  
- Scoring & leaderboard to track performance  

---

## Demo

> *Insert screenshots / GIFs here displaying the game in action — e.g., webcam overlay, scoring, leaderboard.*

---

## Architecture & Design

- **Data layer**: Pose data collected and stored for each dance move.  
- **Model**: Simple supervised classifier that compares input pose vectors to target move prototypes.  
- **Real-time loop**: Capture video frame → detect pose → compute similarity → overlay visuals → update score.  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- Packages: `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`, etc.  
- A webcam (or video input)  

### Installation

```bash
git clone https://github.com/disisid17/DanceOff.git
cd DanceOff
pip install -r requirements.txt
