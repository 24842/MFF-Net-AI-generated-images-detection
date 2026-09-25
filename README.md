# MFF-Net-AI-generated-images-detection
## 📖Introduction
This repository contains the implementation of the paper: "MFF-Net: Multi-scale Feature Fusion Network for Universal AI-generated Image Detection"

### Abstract:
With the rapid advancement of generative models, the emergence of high-fidelity synthetic images has raised significant security concerns regarding disinformation and forgery, leading to the demand for detectors capable 
of distinguishing between AI-generated fake images and real images. The cues for distinguishing AI-generated images from real ones may appear from various aspects, like low-level artifacts, high-level semantics, etc. How
ever, existing detectors often rely on single-view prior information or image features, leading to sub-optimal performance and limited robustness when facing cross-generator image detection. To address these issues, we 
propose a Multi-view Feature Fusion Network (MFF-Net) to take advantage of image features from the high level semantics view and the high-low frequency view for AI-generated image detection. Specifically, to capture 
the high-level semantics, we utilize pre-trained DINOv3 to compute the visual embedding, which can enable the model to discern AI-generated images based on semantics and contextual information. To capture the robust low
level artifacts, we compute low-level features by extracting high-frequency features and low-frequency features from multiple sub-views, including original noise patterns and two types of errors, that is, the reconstruction error and perception error between the original image and its reconstruction counterpart. These multi-view features are dynamically integrated via a Gated Transformer to distinguish AI-generated images from real ones. Extensive experiments on 4 benchmarks, AIGCDetectBenchmark, GenImage, Chameleon , and DFBench, demonstrate that MFF-Net achieves state-of-the-art performance and exhibits better generalization in cross-generator scenarios, while maintaining a relatively good reasoning efficiency compared to existing detectors.

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

### 🛠️Setup
First, clone the repository locally.
```bash
git clone git@github.com:24842/MFF-Net-AI-generated-images-detection.git
```
Then, install the necessary packages and pycocotools.
```bash
pip install -r requirement.txt
```
### Dataset
Training set: [CNNspot](https://github.com/peterwang512/CNNDetection) and [GenImage](https://github.com/Andrew-Zhu/GenImage).

Test set: [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file), [GenImage](https://github.com/Andrew-Zhu/GenImage), [DFBench](https://github.com/IntMeGroup/DFBench) and [Chameleon](https://drive.google.com/file/d/1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH/view).

### Usage
#### Step 1: Feature Extraction
First, run the following scripts located in the tools directory to obtain high/low-frequency features, semantic features, and labels:
```bash
python tools/extract_feature.py
```
```bash
python tools/extract_semantic.py
```
```bash
python tools/make_lable.py
```
#### Step 2: Training
```bash
bash /scripts/train.sh
```
#### Step 3: Testing
Test AIGCDetectBenchmark, GenImage and DFBench 
```bash
bash /scripts/test.sh
```
Test Chameleon
```bash
python test_Chaemleon.py
```
### Checkpoints
Our training checkpoints can be downloaded from [link](https://pan.baidu.com/s/1zBYtDykr8PzE9ORoJg9lnA?pwd=mffn).




