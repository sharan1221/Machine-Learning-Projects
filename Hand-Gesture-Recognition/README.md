# Hand Gesture Recognition — SVM + HOG Features

Built an SVM model with HOG feature extraction to classify 
10 different hand gestures from image data.

## Results
- Classes  : 10 gestures (Palm, Fist, Thumb, Index, OK, Down, L, C, and more)
- Features : HOG (Histogram of Oriented Gradients) — captures edges and shapes
- Dataset  : LeapGestRecog (Kaggle)

## What I did
- Loaded thousands of hand gesture images across 10 subjects
- Preprocessed images using OpenCV (resize, grayscale)
- Extracted HOG features instead of raw pixels for better accuracy
- Encoded 10 gesture labels using LabelEncoder
- Trained SVM with RBF kernel using One-vs-Rest strategy
- Evaluated using accuracy, 10x10 confusion matrix, per-class report
- Built interactive predictor showing top 3 gesture predictions

## Key Concept — Why HOG over raw pixels?
HOG captures edges and shapes in the image rather than raw brightness.
Edges define gestures far better — a fist has curved edges, 
a pointing finger has straight vertical edges.

## Tech Stack
Python · scikit-learn · OpenCV · scikit-image · NumPy · matplotlib

## How to Run
```bash
pip install numpy scikit-learn matplotlib opencv-python scikit-image
python hand_gesture_recognition.py
```
Download dataset from:
https://www.kaggle.com/datasets/gti-upm/leapgestrecog
