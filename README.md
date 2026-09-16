# Oxford-IIIT Pet Classification: CNN Architectures from Scratch

A deep-dive computer vision project implementing, training, and evaluating custom Convolutional Neural Networks (CNNs) using PyTorch. This repository explores the mathematical foundations of spatial convolutions and scales up to training classic deep learning architectures on the 37-class Oxford-IIIT Pet Dataset.

---

##  Repository Structure

This project is broken down into four progressive Jupyter Notebooks, moving from fundamental tensor mathematics to full model evaluation.

### 1. `CNN_Foundation.ipynb`
Explores the mathematical and spatial mechanics of convolutions before relying on high-level PyTorch abstractions.
* **Topics Covered:** 
  * Image tensor representations (CHW vs. HWC).
  * The mathematics of sliding windows, kernels, padding, and stride.
  * Multi-channel (3D) convolution operations on RGB image arrays.

### 2. `CNNs.ipynb`
Focuses on dataset preparation and building the first end-to-end baseline models.
* **Topics Covered:**
  * Ingesting the `torchvision.datasets.OxfordIIITPet` dataset.
  * Constructing a `BaselineCNN` using PyTorch's `nn.Conv2d` layers.
  * Defining the forward pass, loss functions (`CrossEntropyLoss`), and optimizers.

### 3. `Deeper_CNN_Architectures.ipynb`
Scales the complexity by implementing historical network architectures and managing custom training loops across CPU/GPU hardware.
* **Topics Covered:**
  * Implementing a **LeNet-like** CNN for hierarchical feature extraction.
  * Implementing an **AlexNet-mini** architecture designed for modern dataset constraints.
  * Executing dynamic training loops with epoch tracking, validation loss monitoring, and dynamic state dictionary (`state_dict`) saving to prevent overfitting.

### 4. `Evaluation_metrics.ipynb`
Analyzes the performance of the trained weights through statistical and visual metrics.
* **Topics Covered:**
  * Plotting Training vs. Validation Loss curves using Matplotlib to diagnose model variance/bias.
  * Generating a 37x37 class-level heatmap via a Scikit-Learn/Seaborn Confusion Matrix to identify specific breed misclassifications.
  * Safely loading serialized `.pth` weights into evaluation environments.

---

## Tech Stack & Engineering Practices
* **Core Framework:** PyTorch, Torchvision
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib, Seaborn, Scikit-learn (Metrics)
* **Workflow:** Hardware-agnostic training (`cuda`/`cpu` dynamic routing), safe model serialization (`state_dict` vs full pickle).

---

##  Getting Started

### Prerequisites
To run these notebooks locally, ensure you have the following installed:
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
