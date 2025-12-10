from DeepMachineT import DeepMachineT
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==================== UPLOADING DATA ====================
print("Загрузка данных из BreastCancerWisconsin.csv...")
df = pd.read_csv('BreastCancerWisconsin.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

# Target variable: 0 - benign (B), 1 - malignant (M)
diagnosis_map = {'B': 0, 'M': 1}
y_target = df['diagnosis'].map(diagnosis_map).tolist()

# Binarization of features (10 main 'mean' features)
thresholds = {
    'radius_mean': (12.22, 14.81),
    'texture_mean': (17.16, 20.76),
    'perimeter_mean': (78.35, 96.41),
    'area_mean': (461.18, 674.15),
    'smoothness_mean': (0.089, 0.102),
    'compactness_mean': (0.073, 0.116),
    'concavity_mean': (0.038, 0.103),
    'concave points_mean': (0.024, 0.059),
    'symmetry_mean': (0.167, 0.191),
    'fractal_dimension_mean': (0.059, 0.064)
}

binary_features = []
for idx, row in df.iterrows():
    fv = []
    for feature_name, (low_thresh, high_thresh) in thresholds.items():
        value = row[feature_name]
        fv.extend([
            1 if value < low_thresh else 0,
            1 if low_thresh <= value < high_thresh else 0,
            1 if value >= high_thresh else 0
        ])
    binary_features.append(fv)

# Train/Test Split (80/20)
indices_by_class = {0: [], 1: []}
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

machine = DeepMachineT(num_features=30, depth_memory=5, layers=[20, 2])
epochs = 100
print(f"\nStart of training ({epochs} epochs)...")

# Перемешивание
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
        print(f"Epoch {epoch:3d}: {acc:.2f}%")

print(f"\nThe finish line! Final accuracy: {accuracy_list[-1]:.2f}%")

# ==================== ГРАФИК ====================
accuracy_smooth = gaussian_filter1d(accuracy_list, sigma=3)

plt.figure(figsize=(10, 5))
plt.plot(epoch_list, accuracy_smooth, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuaracy, %")
plt.title("Dynamics of DeepMachineT training accuracy on Breast Cancer Wisconsin dataset")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
