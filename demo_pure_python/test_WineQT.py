from DeepMachineT import DeepMachineT
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# ==================== UPLOADING DATA ====================
print("Загрузка данных из WineQT.csv...")
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'datasets', 'WineQT.csv'))

# Target variable - wine quality
# 0 - low (3-4), 1 - medium (5-6), 2 - high (7-8)
def map_quality(q):
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2

y_target = df['quality'].apply(map_quality).tolist()

# Binarization
thresholds = {
    'fixed acidity': (7.4, 8.7),
    'volatile acidity': (0.43, 0.60),
    'citric acid': (0.14, 0.36),
    'residual sugar': (2.0, 2.4),
    'chlorides': (0.074, 0.085),
    'free sulfur dioxide': (9.0, 18.0),
    'total sulfur dioxide': (26.0, 51.0),
    'density': (0.996, 0.997),
    'pH': (3.24, 3.37),
    'sulphates': (0.57, 0.68),
    'alcohol': (9.7, 10.8)
}

binary_features = []
for idx, row in df.iterrows():
    feature_vector = []
    for feature_name, (low_thresh, high_thresh) in thresholds.items():
        value = row[feature_name]
        feature_vector.append(1 if value < low_thresh else 0)
        feature_vector.append(1 if low_thresh <= value < high_thresh else 0)
        feature_vector.append(1 if value >= high_thresh else 0)
    binary_features.append(feature_vector)

# Train/Test Split (80/20)
indices_by_class = {0: [], 1: [], 2: []}
for i, label in enumerate(y_target):
    indices_by_class[label].append(i)

train_indices, test_indices = [], []
for label in indices_by_class:
    random.shuffle(indices_by_class[label])
    split = int(len(indices_by_class[label]) * 0.8)
    train_indices.extend(indices_by_class[label][:split])
    test_indices.extend(indices_by_class[label][split:])

X_train = [binary_features[i] for i in train_indices]
y_train = [y_target[i] for i in train_indices]
X_test = [binary_features[i] for i in test_indices]
y_test = [y_target[i] for i in test_indices]

# ==================== TRAINING ====================

machine = DeepMachineT(num_features=33, depth_memory=5, layers=[20, 3])
epochs = 100
print(f"\nStart of training ({epochs} epochs)...")

combined = list(zip(X_train, y_train))
random.shuffle(combined)
X_sh, y_sh = zip(*combined)
X_sh, y_sh = list(X_sh), list(y_sh)

epoch_list = []
accuracy_list = []

for epoch in range(1, epochs):
    machine.learn(X_sh, y_sh, epochs=1)
    
    correct = 0
    for x, y_true in zip(X_test, y_test):
        y_pred, _, _, _ = machine.predict(x)
        if y_pred == y_true:
            correct += 1
    acc = (correct / len(X_test)) * 100
    
    epoch_list.append(epoch)
    accuracy_list.append(acc)
    
    if epoch % 1 == 0:
        print(f"Rpoch {epoch:3d}: {acc:.2f}%")

print(f"\nThe finish line! Final accuracy: {accuracy_list[-1]:.2f}%")

# ==================== GRAPH OUTPUT ====================
accuracy_smooth = gaussian_filter1d(accuracy_list, sigma=3)

plt.figure(figsize=(10, 5))
plt.plot(epoch_list, accuracy_smooth, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuaracy, %")
plt.title("Dynamics of DeepMachineT training accuracy on WineQT dataset")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

