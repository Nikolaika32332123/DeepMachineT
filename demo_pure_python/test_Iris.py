from DeepMachineT import DeepMachineT
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os 

# ==================== UPLOADING DATA ====================
print("Загрузка данных из Iris.csv...")
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, 'datasets', 'Iris.csv'))

# Target variable: 0-Iris-setosa, 1-Iris-versicolor, 2-Iris-virginica
species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_target = df['Species'].map(species_map).tolist()

# Binarization
binary_features = []
for idx, row in df.iterrows():
    fv = []
    # SepalLengthCm
    sl = row['SepalLengthCm']
    fv.extend([1 if sl < 5.5 else 0, 1 if 5.5 <= sl < 6.5 else 0, 1 if sl >= 6.5 else 0])
    # SepalWidthCm
    sw = row['SepalWidthCm']
    fv.extend([1 if sw < 3.0 else 0, 1 if 3.0 <= sw < 3.5 else 0, 1 if sw >= 3.5 else 0])
    # PetalLengthCm
    pl = row['PetalLengthCm']
    fv.extend([1 if pl < 2.5 else 0, 1 if 2.5 <= pl < 5.0 else 0, 1 if pl >= 5.0 else 0])
    # PetalWidthCm
    pw = row['PetalWidthCm']
    fv.extend([1 if pw < 0.8 else 0, 1 if 0.8 <= pw < 1.8 else 0, 1 if pw >= 1.8 else 0])
    binary_features.append(fv)

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

machine = DeepMachineT(num_features=12, depth_memory=5, layers=[20, 3])
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
        y_pred, _, _,_ = machine.predict(x)
        if y_pred == y_true:
            correct += 1
    acc = (correct / len(X_test)) * 100
    
    epoch_list.append(epoch)
    accuracy_list.append(acc)
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch:3d}: {acc:.2f}%")

print(f"\nThe finish line! Final accuracy: {accuracy_list[-1]:.2f}%")

# ==================== GRAPH OUTPUT ====================
accuracy_smooth = gaussian_filter1d(accuracy_list, sigma=3)

plt.figure(figsize=(10, 5))
plt.plot(epoch_list, accuracy_smooth, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuaracy, %")
plt.title("Dynamics of DeepMachineT training accuracy on IRIS dataset")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
