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
