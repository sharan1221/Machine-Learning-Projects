import os                                    
import numpy as np                           
import cv2                                   
from skimage.feature import hog             
from sklearn.svm import SVC                  
from sklearn.model_selection import train_test_split   
from sklearn.metrics import (accuracy_score,          
                              classification_report,   
                              confusion_matrix)       
from sklearn.preprocessing import StandardScaler, LabelEncoder 
import matplotlib                          
matplotlib.use('Agg')
import matplotlib.pyplot as plt            
import warnings
warnings.filterwarnings('ignore')

IMAGE_SIZE   = 64       
MAX_PER_CLASS = 200      
DATA_DIR     = 'leapGestRecog'   

GESTURE_FOLDERS = {
    '01_palm'       : 'Palm',
    '02_l'          : 'L Shape',
    '03_fist'       : 'Fist',
    '04_fist_moved' : 'Fist Moved',
    '05_thumb'      : 'Thumb',
    '06_index'      : 'Index',
    '07_ok'         : 'OK',
    '08_palm_moved' : 'Palm Moved',
    '09_c'          : 'C Shape',
    '10_down'       : 'Down'
}

print("=" * 62)
print("   TASK-04: HAND GESTURE RECOGNITION — SVM + HOG")
print("=" * 62)

if not os.path.exists(DATA_DIR):
    print(f"\n ERROR: '{DATA_DIR}' folder not found!")
    print("   Make sure leapGestRecog/ is in the same folder as this script.")
    print("   Download: https://www.kaggle.com/datasets/gti-upm/leapgestrecog")
    exit()

subject_folders = sorted([
    f for f in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, f))
])
print(f"\n Dataset found: {len(subject_folders)} subjects → {subject_folders}")

def extract_hog_features(image):

    features = hog(
        image,
        orientations=9,       
        pixels_per_cell=(8,8), 
        cells_per_block=(2,2), 
        block_norm='L2-Hys',   
        visualize=False        
    )
    return features            

def load_all_gestures(data_dir, gesture_folders, subject_folders, max_per_class):
    X = []        
    y = []        
    counts = {}   

    print(f"\nLoading images from all subjects and gestures...")

    for gesture_folder, gesture_name in gesture_folders.items():
        gesture_count = 0
        counts[gesture_name] = 0
        for subject in subject_folders:

            if gesture_count >= max_per_class:
                break
            gesture_path = os.path.join(data_dir, subject, gesture_folder)
            
            if not os.path.exists(gesture_path):
                continue
            
            for filename in os.listdir(gesture_path):

                if gesture_count >= max_per_class:
                    break

                img_path = os.path.join(gesture_path, subject, filename)
                
                if not os.path.exists(img_path):
                    img_path = os.path.join(gesture_path, filename)

                if not os.path.exists(img_path):
                    continue

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    features = extract_hog_features(img)

                    X.append(features)
                    y.append(gesture_name)
                    gesture_count += 1
                    counts[gesture_name] += 1

                except Exception:
                    continue

        print(f"  {gesture_name:15s}: {counts[gesture_name]} images loaded")

    return np.array(X), np.array(y)

X, y_raw = load_all_gestures(DATA_DIR, GESTURE_FOLDERS, subject_folders, MAX_PER_CLASS)

print(f"\n Total images loaded : {X.shape[0]}")
print(f"   Features per image  : {X.shape[1]} (HOG features)")
print(f"   Gesture classes     : {len(np.unique(y_raw))}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = list(le.classes_)

print(f"\n Gesture labels encoded:")
for i, name in enumerate(class_names):
    count = (y == i).sum()
    print(f"   {i:2d} → {name:15s} ({count} images)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n HOG features scaled (StandardScaler)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y       
)

print(f"\n Data split:")
print(f"   Training samples : {len(X_train)} (80%)")
print(f"   Testing samples  : {len(X_test)}  (20%)")

print(f"\n Training SVM model on {len(X_train)} images...")
print(f"   Kernel: RBF  |  C=10  |  gamma=scale")
print(f"   (This may take 3-8 minutes — please wait...)")

svm_model = SVC(
    kernel='rbf',        
    C=10,                
    gamma='scale',       
    probability=True,    
    random_state=42,
    decision_function_shape='ovr'  
)
svm_model.fit(X_train, y_train)
print(f" Model trained successfully!")

y_pred   = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred, target_names=class_names)

print(f"\n{'='*62}")
print(f"   MODEL PERFORMANCE")
print(f"{'='*62}")
print(f"   Overall Accuracy : {accuracy*100:.2f}%")
print(f"\n   Per-class accuracy:")
for i, name in enumerate(class_names):
    class_correct = cm[i][i]
    class_total   = cm[i].sum()
    class_acc     = class_correct / class_total * 100 if class_total > 0 else 0
    bar = '█' * int(class_acc // 5)
    print(f"   {name:15s}: {class_acc:5.1f}%  {bar}")

print(f"\n   Full Classification Report:")
print(report)

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('Hand Gesture Recognition — SVM + HOG Features',
             fontsize=15, fontweight='bold', y=0.98)

im = axes[0,0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0,0].set_title('Confusion Matrix (10 Gestures)', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Predicted Gesture')
axes[0,0].set_ylabel('Actual Gesture')
axes[0,0].set_xticks(range(len(class_names)))
axes[0,0].set_yticks(range(len(class_names)))
axes[0,0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
axes[0,0].set_yticklabels(class_names, fontsize=8)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        val = cm[i][j]
        color = 'white' if val > cm.max()/2 else 'black'
        axes[0,0].text(j, i, str(val), ha='center', va='center',
                       fontsize=7, color=color, fontweight='bold')
plt.colorbar(im, ax=axes[0,0])

per_class_acc = []
for i in range(len(class_names)):
    total   = cm[i].sum()
    correct = cm[i][i]
    per_class_acc.append((correct / total * 100) if total > 0 else 0)

colors_bar = plt.cm.RdYlGn(np.array(per_class_acc) / 100)
bars = axes[0,1].barh(class_names, per_class_acc,
                       color=colors_bar, edgecolor='black', alpha=0.85)
axes[0,1].set_xlabel('Accuracy (%)')
axes[0,1].set_title('Per-Gesture Accuracy', fontsize=12, fontweight='bold')
axes[0,1].set_xlim(0, 115)
axes[0,1].axvline(x=accuracy*100, color='navy', lw=2,
                   linestyle='--', label=f'Overall: {accuracy*100:.1f}%')
axes[0,1].legend(fontsize=9)
axes[0,1].grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, per_class_acc):
    axes[0,1].text(val + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')
    
axes[1,0].set_title('Sample Test Predictions', fontsize=12, fontweight='bold')
axes[1,0].axis('off')
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
for idx, sample_idx in enumerate(sample_indices):
    ax_img = axes[1,0].inset_axes([
        (idx % 5) * 0.20, 0.5 - (idx // 5) * 0.52, 0.18, 0.46
    ])
    hog_img = scaler.inverse_transform(X_test[sample_idx].reshape(1, -1))
    vis_size = int(np.sqrt(min(IMAGE_SIZE*IMAGE_SIZE, hog_img.shape[1])))
    display_img = hog_img.reshape(-1)[:vis_size*vis_size].reshape(vis_size, vis_size)
    ax_img.imshow(display_img, cmap='gray')
    actual    = class_names[y_test[sample_idx]]
    predicted = class_names[y_pred[sample_idx]]
    correct   = actual == predicted
    color     = 'limegreen' if correct else 'red'
    ax_img.set_title(f'A:{actual[:6]}\nP:{predicted[:6]}',
                      fontsize=6.5, color=color, fontweight='bold')
    ax_img.axis('off')
proba = svm_model.predict_proba(X_test)
max_confidence = np.max(proba, axis=1) * 100
correct_mask   = y_test == y_pred
axes[1,1].hist(max_confidence[correct_mask], bins=20, alpha=0.65,
               color='#2ECC71', edgecolor='black', label='Correct predictions')
axes[1,1].hist(max_confidence[~correct_mask], bins=20, alpha=0.65,
               color='#E74C3C', edgecolor='black', label='Wrong predictions')
axes[1,1].axvline(x=50, color='black', lw=2, linestyle='--', label='50% threshold')
axes[1,1].set_xlabel('Model Confidence (%)')
axes[1,1].set_ylabel('Number of Images')
axes[1,1].set_title('Confidence: Correct vs Wrong Predictions',
                     fontsize=12, fontweight='bold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gesture_results.png', dpi=150, bbox_inches='tight')
print(f"\n Charts saved as 'gesture_results.png'")

print(f"\n{'='*62}")
print(f"   CLASSIFY A NEW HAND GESTURE IMAGE")
print(f"{'='*62}")
print(f"\n   Gesture classes the model knows:")
for i, name in enumerate(class_names):
    print(f"   {i:2d}. {name}")

print(f"\n   Enter the full path to a hand gesture image.")
print(f"   Example: C:\\Users\\YourName\\Desktop\\gesture.png")
print(f"   (Press Enter to skip)\n")

img_path = input("   Image path: ").strip().strip('"').strip("'")

if img_path == '':
    print("\n   Skipped. Program complete!")
else:
    if not os.path.exists(img_path):
        print(f"\n File not found: {img_path}")
    else:
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(" Could not read image file.")
            else:
                img_resized  = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img_gray     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                img_hog      = extract_hog_features(img_gray)   # HOG features
                img_scaled   = scaler.transform(img_hog.reshape(1, -1))

                prediction   = svm_model.predict(img_scaled)[0]
                confidence   = svm_model.predict_proba(img_scaled)[0]
                gesture_name = class_names[prediction]
                conf_percent = confidence[prediction] * 100

                top3_idx  = np.argsort(confidence)[::-1][:3]

                print(f"\n{'='*62}")
                print(f"   RESULT")
                print(f"{'='*62}")
                print(f"    Detected Gesture : {gesture_name}")
                print(f"    Confidence        : {conf_percent:.1f}%")
                print(f"\n   Top 3 predictions:")
                for rank, idx in enumerate(top3_idx, 1):
                    print(f"   {rank}. {class_names[idx]:15s}: {confidence[idx]*100:.1f}%")
                print(f"\n   Model overall accuracy: {accuracy*100:.2f}%")
                print(f"{'='*62}")

        except Exception as e:
            print(f"Error: {e}")

print(f"\nDone! Check 'gesture_results.png' for all charts.")
print("=" * 62)
