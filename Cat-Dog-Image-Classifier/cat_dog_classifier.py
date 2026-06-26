
import os                          
import numpy as np                 
import cv2                        
from sklearn.svm import SVC       
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from sklearn.preprocessing import StandardScaler       
import matplotlib                
matplotlib.use('Agg')
import matplotlib.pyplot as plt   
import warnings                   
warnings.filterwarnings('ignore')

IMAGE_SIZE   = 64      
MAX_IMAGES   = 1000    
DATA_DIR     = 'PetImages'   

print("=" * 60)
print("   TASK-03: CAT vs DOG CLASSIFIER — SVM")
print("=" * 60)

if not os.path.exists(DATA_DIR):
    print(f"\n ERROR: '{DATA_DIR}' folder not found!")
    print("   Make sure PetImages/ is in the same folder as this script.")
    print("   Download from: https://www.microsoft.com/en-us/download/details.aspx?id=54765")
    exit()

cat_dir = os.path.join(DATA_DIR, 'Cat')
dog_dir = os.path.join(DATA_DIR, 'Dog')

if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
    print("ERROR: Could not find Cat/ or Dog/ inside PetImages/")
    exit()

print(f"\nDataset folder found: {DATA_DIR}/")
print(f"   Cat folder: {len(os.listdir(cat_dir))} files")
print(f"   Dog folder: {len(os.listdir(dog_dir))} files")

def load_images(folder_path, label, max_count):
    images = []   
    labels = []   
    loaded = 0    

    all_files = os.listdir(folder_path)

    for filename in all_files:
        if loaded >= max_count:
            break
        img_path = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flat = img.flatten()
            images.append(img_flat)
            labels.append(label)
            loaded += 1

        except Exception:
            continue

    return images, labels

print(f"\n Loading cat images (max {MAX_IMAGES})...")
cat_images, cat_labels = load_images(cat_dir, label=0, max_count=MAX_IMAGES)
print(f"    Loaded {len(cat_images)} cat images")

print(f"\n Loading dog images (max {MAX_IMAGES})...")
dog_images, dog_labels = load_images(dog_dir, label=1, max_count=MAX_IMAGES)
print(f"   Loaded {len(dog_images)} dog images")

X = np.array(cat_images + dog_images)   
y = np.array(cat_labels + dog_labels)  

print(f"\n Combined dataset:")
print(f"   Total images : {X.shape[0]}")
print(f"   Features per image : {X.shape[1]} pixels (64x64 flattened)")
print(f"   Cats (label 0) : {(y == 0).sum()}")
print(f"   Dogs (label 1) : {(y == 1).sum()}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nPixel values scaled (0-255 → normalised)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData split:")
print(f"   Training images : {len(X_train)} (80%)")
print(f"   Testing images  : {len(X_test)} (20%)")

print(f"\n Training SVM model (this may take 2-5 minutes)...")
print(f"   Kernel: RBF  |  C=10  |  gamma=scale")

svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

print(f"SVM model trained successfully!")
print(f"   Support vectors found: {sum(svm_model.n_support_)} total")
print(f"   - Cat support vectors: {svm_model.n_support_[0]}")
print(f"   - Dog support vectors: {svm_model.n_support_[1]}")

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred, target_names=['Cat', 'Dog'])

print(f"\n{'='*60}")
print(f"   MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"   Accuracy : {accuracy*100:.2f}%")
print(f"\n   Confusion Matrix:")
print(f"                Predicted Cat  Predicted Dog")
print(f"   Actual Cat        {cm[0][0]:4d}           {cm[0][1]:4d}")
print(f"   Actual Dog        {cm[1][0]:4d}           {cm[1][1]:4d}")
print(f"\n   Classification Report:")
print(report)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Cat vs Dog Classifier — SVM Results', fontsize=15, fontweight='bold')

im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Predicted Label'); axes[0, 0].set_ylabel('Actual Label')
axes[0, 0].set_xticks([0, 1]); axes[0, 0].set_yticks([0, 1])
axes[0, 0].set_xticklabels(['Cat', 'Dog']); axes[0, 0].set_yticklabels(['Cat', 'Dog'])
for i in range(2):
    for j in range(2):
        axes[0, 0].text(j, i, str(cm[i][j]), ha='center', va='center',
                        fontsize=22, fontweight='bold',
                        color='white' if cm[i][j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=axes[0, 0])

categories  = ['Cat Accuracy', 'Dog Accuracy', 'Overall Accuracy']
cat_acc = cm[0][0] / (cm[0][0] + cm[0][1])
dog_acc = cm[1][1] / (cm[1][0] + cm[1][1])
values  = [cat_acc * 100, dog_acc * 100, accuracy * 100]
colors2 = ['#E74C3C', '#3498DB', '#2ECC71']
bars = axes[0, 1].bar(categories, values, color=colors2, edgecolor='black', alpha=0.85)
axes[0, 1].set_ylim(0, 110)
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Accuracy Breakdown', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, f'{val:.1f}%',
                    ha='center', fontweight='bold', fontsize=11)

axes[1, 0].axis('off')
axes[1, 0].set_title('Sample Predictions (8 test images)', fontsize=12, fontweight='bold')
label_names = {0: 'Cat', 1: 'Dog'}
sample_indices = np.random.choice(len(X_test), 8, replace=False)
grid = axes[1, 0].inset_axes([0, 0, 1, 0.95])
grid.axis('off')

for idx, sample_idx in enumerate(sample_indices):
    ax_img = axes[1, 0].inset_axes([
        (idx % 4) * 0.25, 0.5 - (idx // 4) * 0.5, 0.23, 0.45
    ])

    img_display = scaler.inverse_transform(X_test[sample_idx].reshape(1, -1))
    img_display = img_display.reshape(IMAGE_SIZE, IMAGE_SIZE)
    ax_img.imshow(img_display, cmap='gray')
    actual    = label_names[y_test[sample_idx]]
    predicted = label_names[y_pred[sample_idx]]
    correct   = actual == predicted
    color     = 'green' if correct else 'red'
    ax_img.set_title(f'A:{actual}\nP:{predicted}',
                     fontsize=7, color=color, fontweight='bold')
    ax_img.axis('off')

proba = svm_model.predict_proba(X_test)
dog_confidence   = proba[y_pred == 1, 1]
cat_confidence   = proba[y_pred == 0, 0]
axes[1, 1].hist(cat_confidence, bins=20, alpha=0.6, color='#E74C3C',
                label='Cat predictions', edgecolor='black')
axes[1, 1].hist(dog_confidence, bins=20, alpha=0.6, color='#3498DB',
                label='Dog predictions', edgecolor='black')
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', lw=2, label='50% threshold')
axes[1, 1].set_xlabel('Prediction Confidence')
axes[1, 1].set_ylabel('Number of Images')
axes[1, 1].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cat_dog_results.png', dpi=150, bbox_inches='tight')
print(f"\nCharts saved as 'cat_dog_results.png'")

print(f"\n{'='*60}")
print(f"   CLASSIFY YOUR OWN IMAGE")
print(f"{'='*60}")
print(f"\n   Enter the full path to any cat or dog image.")
print(f"   Example: C:\\Users\\YourName\\Desktop\\mycat.jpg")
print(f"   (or press Enter to skip)\n")

img_path = input("   Image path: ").strip().strip('"').strip("'")

if img_path == '':
    print("\n   Skipped. Program complete!")
else:
    if not os.path.exists(img_path):
        print(f"\nFile not found: {img_path}")
    else:
        try:
            img = cv2.imread(img_path)

            if img is None:
                print(" Could not read the image file.")
            else:
               
                img_resized  = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img_gray     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                img_flat     = img_gray.flatten().reshape(1, -1)
                img_scaled   = scaler.transform(img_flat)

              
                prediction   = svm_model.predict(img_scaled)[0]
                confidence   = svm_model.predict_proba(img_scaled)[0]
                label        = 'DOG ' if prediction == 1 else 'CAT '
                conf_percent = confidence[prediction] * 100

                print(f"\n{'='*60}")
                print(f"   RESULT FOR YOUR IMAGE")
                print(f"{'='*60}")
                print(f"   Prediction  : {label}")
                print(f"   Confidence  : {conf_percent:.1f}%")
                print(f"   Cat probability : {confidence[0]*100:.1f}%")
                print(f"   Dog probability : {confidence[1]*100:.1f}%")
                print(f"   Model accuracy  : {accuracy*100:.2f}% on test set")
                print(f"{'='*60}")

        except Exception as e:
            print(f" Error processing image: {e}")

print(f"\n Done! Check 'cat_dog_results.png' for all charts.")
print("=" * 60)