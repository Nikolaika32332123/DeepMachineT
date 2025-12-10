import os
from dmt_package.dmt import DMT

# =========================================================================
# 1. Initialization
# =========================================================================
print("--- 1. DMT Initialization ---")
# depth_memory=5: memory depth (synapse strength)
# num_features=5: input vector length
# layers=[50, 20, 2]: two hidden layers and an output layer for 2 classes
model = DMT(depth_memory=5, num_features=5, layers=[50, 20, 2])
print("Model created successfully.")


# =========================================================================
# 2. Supervised Learning (SL)
# =========================================================================
print("\n--- 2. Supervised Learning (SL) ---")
# Simple dataset (XOR-like + noise)
X_train = [
    [1, 1, 1, 0, 0],  # Class 0
    [0, 1, 0, 1, 1],  # Class 1
    [1, 1, 0, 1, 0],  # Class 0
    [0, 1, 1, 0, 1],  # Class 1
    [1, 0, 0, 0, 0],  # Class 0
    [0, 0, 1, 1, 1]   # Class 1
]
y_train = [0, 1, 0, 1, 0, 1]

model.learn_sl(X_train, y_train, epochs=300, verbose=True)

# Check accuracy on the training set
accuracy = model.evaluate(X_train, y_train)
print(f"Accuracy after SL: {accuracy:.2f}%")


# =========================================================================
# 3. Reinforcement Learning (RL)
# =========================================================================
print("\n--- 3. Reinforcement Learning (RL) ---")
# Fine-tune the model on a single example
rl_sample = [1, 1, 1, 0, 0]  # This should be Class 0
# Predict before fine-tuning
pred_before = model.predict(rl_sample)
print(f"Prediction before RL: {pred_before}")

# Give positive reinforcement (+1) if the model is correct,
# or negative (-1) to correct it if it made a mistake.
model.learn_rl(rl_sample, reward=1)
print("RL update completed.")


# =========================================================================
# 4. Save & Load (Binary & JSON)
# =========================================================================
print("\n--- 4. Save & Load ---")
# Binary format (fast)
model.save("my_model.bin")
print("Saved to .bin")

# JSON format (human-readable)
model.save("my_model_export.json", as_json=True)
print("Saved to .json")

# Loading back from .bin
abs_path = os.path.abspath("my_model.bin")
model.load(abs_path)
print("Model loaded from .bin")

# Check that nothing broke
acc_after_load = model.evaluate(X_train, y_train)
print(f"Accuracy after load: {acc_after_load:.2f}%")


# =========================================================================
# 5. Model Pruning (Compression)
# =========================================================================
print("\n--- 5. Model Pruning ---")
# Removes inactive neurons and saves a "clean" architecture
model.compress_pruning("my_model_pruned.json")
# "Compressed model saved..." message and stats are printed inside the method


# =========================================================================
# 6. Inference (Prediction)
# =========================================================================
print("\n--- 6. Inference ---")
test_sample = [0, 1, 0, 1, 1]
prediction = model.predict(test_sample)
print(f"Input: {test_sample} -> Predicted Class: {prediction}")
