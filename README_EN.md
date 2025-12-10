# DeepMachineT: An Energy-Efficient ML Framework in C++

DeepMachineT is a high-performance machine learning framework written in pure C++ with Python bindings. The system is designed to explore alternative learning paradigms, abandoning classical backpropagation in favor of local reinforcement learning and integer arithmetic.

The main idea: maximum efficiency by eliminating expensive floating-point operations and using only integer logic, making the system ideal for embedded systems and energy-critical applications.

---

## 🎯 Key Features

### 1. **Green AI**
- ✅ Only **integer arithmetic** (`uint8` weights)
- ✅ No matrix multiplication (FLOAT operations)
- ✅ Operations: addition, comparison, logical AND
- ✅ Orders of magnitude less power consumption** on CPU vs. PyTorch/TensorFlow

### 2. **Biologically inspired learning**
- ✅ Local learning rules (LTP/LTD) instead of global gradient descent
- ✅ Stochastic hypothesis search** (synaptogenesis)
- ✅ Adaptive plasticity** through the "awareness" mechanism (Deliberate mode)
- ✅ Solution to the problem of **catastrophic forgetting**

### 3. **Full Interpretability**
- ✅ Each decision has an **explicit reasoning trace** — a chain of activated neurons
- ✅ The **system confidence** is known at every point in time
- ✅ **Neurosymbolic architecture** (facts, not abstract features)

### 4. **Built-in model optimization (Pruning)**
- ✅ Automatic removal of **inactive neurons** without loss of accuracy
- ✅ Compression by **50%+ based on the number of weights**
- ✅ Detailed pruning statistics by layer

### 5. **Easy integration**
- ✅ C API for integration into any language (Java, C#, Go, Rust, Node.js, etc.)
- ✅ Python wrapper via `ctypes` (no additional dependencies)
- ✅ Standard save formats (BIN for speed, JSON for debugging)

---

## ⚙️ Installation and Compilation

### Requirements
- **C++17** compiler (g++, clang, or MSVC)
- **AVX2** processor support (recommended)
- **Python 3.7+** (only for using the Python wrapper)

### Compiling on Windows (MinGW)

```bash
cd cpp_core
g++ -O3 -shared -fPIC -mavx2 -o ../bin/dmt_lib.dll dmt_lib.cpp
```

Or use a ready-made script:
```bash
.\compile_windows.bat
```

### Compiling on Linux/macOS

```bash
cd cpp_core
g++ -O3 -shared -fPIC -mavx2 -o ../bin/libdmt.so dmt_lib.cpp
```

Or use a ready-made script:
```bash
chmod +x compile_linux.sh
./compile_linux.sh
```

### Installation check

```python
from wrappers.python.dmt_package.dmt import DMT

# If there are no errors, everything is installed correctly
model = DMT(depth_memory=5, num_features=5, layers=[50, 20, 2])
print("✓ DeepMachineT is ready to go!")
```

---

## 🚀 Quick Start

### Simple example (Python)

```python
from dmt_package.dmt import DMT

# 1. Creating a model
model = DMT(depth_memory=5, num_features=5, layers=[50, 20, 2])

# 2. Data Preparation (binary vectors 0 and 1)
X_train = [
[1, 1, 1, 0, 0], # Sample of class 0
[0, 1, 0, 1, 1], # Sample of class 1
[1, 1, 0, 1, 0], # Sample of class 0
[0, 1, 1, 0, 1], # Sample of class 1
]
y_train = [0, 1, 0, 1]

# 3. Training (local training, WITHOUT gradient descent)
model.learn_sl(X_train, y_train, epochs=300, verbose=True)

# 4. Accuracy Evaluation
accuracy = model.evaluate(X_train, y_train)
print(f"Accuracy: {accuracy: .2f}%")

# 5. Prediction
prediction = model.predict([0, 0, 1, 1, 1])
print(f"Prediction: {prediction}")

# 6. Saving the Model
model.save("my_model.bin")
```

### Usage from Other Languages

DeepMachineT has a **C API** (file `cpp_core/include/dmt_lib.h`), so it can be integrated into:

**Java / Kotlin (JNI):**
```java
System.loadLibrary("dmt_lib");
// ... call C functions
```

**C# / .NET (P/Invoke):**
```csharp
[DllImport("dmt_lib.dll")]
public static extern IntPtr DMT_Create(...);
```

**Go (cgo):**
```go
// #cgo LDFLAGS: -L./bin -ldmt_lib
import "C"
```

---

## 📚 API Documentation (Python)

### Main Class: `DMT`

#### Initialization
```python
model = DMT(depth_memory=5, num_features=5, layers=[50, 20, 2])
```

| Parameter | Type | Description |
|----------|-----|---------|
| `depth_memory` | int | Maximum synapse "memory depth" (1-255). Maximum connection weight. |
| `num_features` | int | Input vector size (number of features) |
| `layers` | List[int] | Network architecture. Example: `[50, 20, 2]` = 50→20→2 neurons |

#### Learning Methods

**`learn_sl(X, y, epochs=100, verbose=True)`** — Supervised Learning
- `X`: Training data (List[List[int]])
- `y`: Class labels (List[int])
- `epochs`: Number of epochs
- `verbose`: Whether to show the progress bar

**`learn_rl(x, reward)`** — Reinforcement Learning
- `x`: Input sample (List[int])
- `reward`: +1 (true) or -1 (false)

#### Main Methods

**`predict(x) -> int`** — Class prediction
**`evaluate(X, y) -> float`** — Accuracy estimate (in percent)
**`save(path, as_json=False)`**— Saving weights
**`load(path)`** — Loading weights (both BIN and JSON formats)
**`compress_pruning(save_path)`** — Compression by removing inactive neurons

See `wrappers/python/main.py` for usage examples.

---

## 🧠 Architecture and Operating Principles

### Three Neuron States

| State | Weight | Description |
|-----------|---------|---------|
| **Empty** | w=0 | Connection not formed |
| **Hypothesis** | w=1 | Unstable connection (in search) |
| **Knowledge** | w>1 | Consolidated connection (in use) |

### Training: LTP and LTD

**LTP (Strengthening):** If the answer is correct → the weight increases (w ← w+1)
**LTD (Dampening):** If the answer is incorrect → the weight decreases (w ← w-1)
**Knowledge Protection:** Confident neurons are protected from updating due to errors in other layers

### "Awareness" Mechanism

The system tracks its confidence:
- **Deliberate = 0**: All weights ≥ 1 → confident → learning slows down
- **Deliberate = 1**: Any weight = 1 → uncertain → learning speeds up

This solves the problem of **catastrophic forgetting** and allows the network to adapt the learning rate.

---

## 📊 Experimental Results

| Dataset | DeepMachineT | MLP | Random Forest | SVM |
|---------|--------------|-----|--------------|-----|
| **Iris** | 93.33% | 93.33% | 94.67% | 95.33% |
| **Wine Quality** | 92.70% | 96.08% | 96.63% | 97.76% |
| **Breast Cancer** | 93.33% | 95.26% | 94.20% | 93.15% |

**Key observations:**
- Accuracy **comparable to classical methods**, despite using only a binary reinforcement signal
- Shows **high stability** on medical data
- **Orders of magnitude lower energy consumption** due to integer arithmetic

---

## 🔬 Scientific basis

DeepMachineT is inspired by the ideas of:
- **M.M. Bongard** on concept construction
- **K. Friston's Free Energy Principle** — local energy minimization
- **Synaptic Plasticity** — LTP/LTD in biology
- **Markov Blankets** — locality in neural networks

**For a detailed description of the theory and experiments:** see docs/DeepMachineT_Paper_RU.pdf

---

## 🚀 Future Developments

### Graphical Neural Network Editor (In Progress)

One of the main features of DeepMachineT is its **neurosymbolic architecture**, which makes the network visually interpretable and editable. Development plans include:

- **Visual Network Editor** — a graphical application for visually constructing neural networks
- **AI Assistant** — a built-in assistant that will help design networks and explain decisions

### The project is open to ideas! Potential areas of development

- **Distributed learning** — synchronizing weights between neurons (Gossip algorithm)
- **Multimodality** — integrating different data types (text, images, sound)
- **Language extension** — adaptation to programming languages: C#, Java, JavaScript

---

## 📝 License

MIT License — free to use in commercial and personal projects.

---

## 📧 Questions and Feedback

**Author:** Nikolay Tokarev
**Email:** nikolaos@byte-agi.ru

If you have questions, ideas, or found a bug, let us know!

---

## 📖 Additional Resources

- `cpp_core/dmt_lib.cpp` — C++ kernel source code
- `demo_pure_python/DeepMachineT.py` — Simple Python version for learning
- `wrappers/python/main.py` — Complete examples of all functions
- `docs/DeepMachineT_Paper_RU.pdf` — Articles with theory and experiments

---

**Last updated:** December 10, 2025
**Version:** 1.0-beta
**Status:** Active development