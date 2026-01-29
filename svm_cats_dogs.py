import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

# Folder structure:
# dataset/
#   cats/
#   dogs/

for label, folder in enumerate(['cats', 'dogs']):
       BASE_DIR = os.path.dirname(os.path.abspath(__file__))
       folder_path = os.path.join(BASE_DIR, "dataset", folder)
       for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        data.append(image.flatten())
        labels.append(label)

X = np.array(data)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
