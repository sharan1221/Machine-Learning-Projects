# Cat vs Dog Image Classifier — SVM

Built an SVM model to classify images of cats and dogs 
using pixel data and OpenCV preprocessing.

## Results
- Accuracy : ~72%
- Training : 1,600 images
- Testing  : 400 images
- Classes  : Cat, Dog

## What I did
- Loaded 2,000 real images using OpenCV
- Resized all images to 64x64 pixels
- Converted to grayscale and flattened to 4,096 pixel arrays
- Scaled pixel values using StandardScaler
- Trained SVM with RBF kernel
- Evaluated using accuracy, confusion matrix, classification report
- Built interactive classifier for any new cat or dog image

## Tech Stack
Python · scikit-learn · OpenCV · NumPy · matplotlib

## How to Run
```bash
pip install numpy scikit-learn matplotlib opencv-python
python cat_dog_classifier.py
```
Download dataset from:
https://www.microsoft.com/en-us/download/details.aspx?id=54765
