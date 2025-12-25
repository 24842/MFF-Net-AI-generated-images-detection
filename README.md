# MFF-Net-AI-generated-images-detection
## 📖Introduction
This repository contains the implementation of the paper: "MFF-Net: Multi-scale Feature Fusion Network for Universal AI-generated Image Detection"
### Abstract:
To improve generalization in synthetic image detection, we propose MFF-Net. It integrates seven high- and low-frequency feature groups (via VQ-VAE and VGG16 residual maps) with global semantic features from DINOv3 to capture both frequency anomalies and logical inconsistencies. These multi-source features are dynamically fused using a Gated Transformer encoder. Experimental results on AIGCDetectBenchmark and GenImage demonstrate state-of-the-art (SOTA) performance and robust generalization.

### 👀Method
We propose MFF-Net, a multi-scale fusion framework for universal synthetic image detection. The model employs a hybrid architecture that integrates high/low-frequency features (via image and residual maps) with global semantic features from DINOv3. These multi-source representations are fused through a Gated Attention Transformer within a multi-task learning framework. Finally, an adaptive inference strategy uses a confidence-based gating mechanism to select the most reliable prediction from both specialized experts and fused features.

<div align="center">
  <img src="images/MFF-Net.png" alt="MFF-Net Architecture" width="700">
</div>

### 💻Requirments
We test the codes in the following environments, other versions may also be compatible:
* CUDA 12.1
* Python 3.10.18
* Pytorch 2.5.1

### Setup
First, clone the repository locally.
```bash
git clone 
```
