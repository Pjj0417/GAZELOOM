<div align="center">

# ⚡ GazeLoom ⚡

### Train Driver Gaze Estimation Framework

<img src="https://github.com/user-attachments/assets/3c4055b2-9ec1-40ee-9c79-b664945336d9" width="720" />

<br/>

**A lightweight and robust driver gaze estimation system powered by self-supervised learning and geometry guidance.**

<img src="https://github.com/user-attachments/assets/d8c383f4-6a71-4a12-8f5a-9d0b100ba8db" width="720" />

</div>

---

## 🚀 About

<div align="center">
  <img src="https://github.com/user-attachments/assets/3f2dc547-ade5-4530-8f45-9b555ceeeb4f" width="720" />
</div>

**GazeLoom** is a driver gaze estimation framework designed for intelligent traffic safety and human-vehicle interaction.

It leverages **multi-modal geometric guidance** and **self-supervised feature extraction** to accurately predict driver gaze points in 3D space.

### Highlights

- 🔹 **Lightweight Model** — only **4.97M parameters**
- 🔹 **High-Precision Estimation** — joint prediction of head pose and eye movement
- 🔹 **Real-Time Inference** — suitable for in-vehicle edge deployment
- 🔹 **Strong Generalization** — robust to lighting changes, occlusions, and pose variations
- 🚀 **ONNX Deployment** — Run `onnx.py` to export and deploy GazeLoom in ONNX format. Google Drive: [Download](https://drive.google.com/file/d/1xLQqY6gnw76jP0Xy6-sLKM07R0hgFBw8/view?usp=drive_link)

---


## 🗂️ Data Processing

Before training or evaluation, please download the required datasets and run the corresponding preprocessing scripts.

The preprocessing scripts compile raw annotations into unified JSON files for each split, including head bounding boxes, gaze points, in/out labels, and metadata required by GazeLoom.

### GazeFollow

Download the GazeFollow dataset [here](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset).


python data_prep/preprocess_gazefollow.py \
  --data_path /path/to/gazefollow/data_new
### VideoAttentionTarget

Download the VideoAttentionTarget dataset [here](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset-1).


python data_prep/preprocess_vat.py \
  --data_path /path/to/videoattentiontarget

### ChildPlay

Download the ChildPlay dataset [here](https://www.idiap.ch/en/scientific-research/data/childplay-gaze).


python data_prep/preprocess_childplay.py \
  --data_path /path/to/childplay
### GOO-Real

Download the GOO-Real dataset [here](https://github.com/upeee/GOO-GAZE2021/blob/main/dataset/gooreal-download.txt).


python data_prep/preprocess_goo_real.py \
  --data_path /path/to/goo_real


## 📸 Visuals

<div align="center">

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0f2ecb2d-3cd2-41ab-81fa-27e61aafe383" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/f3e91f3b-5e87-46a1-829a-327f8e3a721d" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/5be57660-463a-4003-a2a8-b5b5d1d75549" width="260"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0fcd3faf-e1b1-49cd-8377-1191fd277ce4" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/9732d414-9a6a-4257-a961-852fca559244" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/1d4f8971-6ece-4c7a-b9c9-ca485f1a7730" width="260"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7548f741-8b97-4528-af2f-92d2100ccae1" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/f996580c-a005-4c7a-b511-fe9c7584b7e4" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/e3bead59-4718-441f-b701-2b23fa40d6dc" width="260"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8c2f4bc4-eb1f-4303-a93b-be26cd17d0f9" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/7302f26f-8360-4d99-9130-3180c2942869" width="260"/></td>
    <td><img src="https://github.com/user-attachments/assets/bb47e194-9054-4cb9-9dc5-e25220f233a8" width="260"/></td>
  </tr>
</table>

</div>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 Geometry-Guided Learning | Combines semantic and geometric priors for robust gaze estimation |
| ⚙️ Self-Supervised Backbone | Reduces dependency on large-scale labeled data |
| 🚗 Driver-Centric Design | Optimized for railway and in-cabin driving environments |
| ⚡ Lightweight Deployment | 4.97M parameters with real-time edge inference capability |

---

## 🧩 Overall Framework

GazeLoom is a lightweight and geometry-guided framework for **3D driver gaze estimation** in railway driving scenarios.

It consists of three key stages:

1. **Feature Extraction**
2. **Geometry Guidance**
3. **Fusion & Prediction**

<div align="center">
  <img src="https://github.com/user-attachments/assets/f19f31bf-c917-4ffc-bb9c-e86c726cc065" width="720" />
</div>

---

## 🎯 Stage 1 — Feature Extraction

A pre-trained self-supervised backbone, **SimDINOv2**, is used to encode driving scene images and generate robust global visual representations.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d655e12a-ee7f-4f0e-91d0-886cc8652a70" width="720" />
</div>

To enhance semantic understanding and spatial perception, GazeLoom incorporates multiple auxiliary cues:

- 🌫️ **Depth Map** — provides 3D structural priors
- ✨ **DISM Saliency Map** — highlights attention-relevant visual regions
- 👤 **Head Pose Features** — offer geometric priors of gaze orientation

These features are fed into the **Multi-modal Geometry Guidance module** for semantic-geometric fusion.

---

## 📐 Stage 2 — Geometry Guidance

The **Multi-modal Geometry Guidance module**, abbreviated as **MGG**, enhances 3D spatial reasoning and structural perception.

### Head Branch

The head branch uses head feature maps with pseudo-heatmap supervision to explicitly model local geometric constraints of gaze direction.

### Depth Branch

The depth branch fuses depth maps and DISM saliency maps to inject global 3D structural priors.

Together, these branches generate a structure-consistent visual-spatial representation for downstream gaze prediction.

---

## 🔗 Stage 3 — Fusion & Prediction

The **Cross-modal Gating Fusion module**, abbreviated as **CGF**, adaptively integrates semantic and spatial features through a gating attention mechanism.

After fusion, the model performs two prediction tasks:

- 🎯 **In-Out Gaze Classification**
- 🔥 **Gaze Heatmap Generation**

The entire model is trained using a multi-task joint optimization framework, improving robustness, generalization, and real-time performance.

---

## 🔍 Module Details

### 🧩 MGG — Multi-modal Geometry Guidance

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d75adaa-3f3c-4131-9506-0fcd5337c5ba" width="720" />
</div>

**MGG** integrates geometric priors from multiple modalities to enhance robustness under complex driving conditions.

#### Input Sources

- Facial landmarks
- Head pose
- Eye-region depth features

#### Core Functions

- Builds multi-modal geometric representations
- Captures spatial relationships between facial structure and orientation
- Applies geometry consistency constraints
- Uses a lightweight transformer to model spatial dependencies

> 💡 MGG helps GazeLoom maintain high precision under lighting changes, head rotations, and partial occlusions.

---

### 🔗 CGF — Cross-modal Gating Fusion

<div align="center">
  <img src="https://github.com/user-attachments/assets/ce87da76-3cdb-48eb-9fb6-aa95cef654af" width="420" />
</div>

**CGF** introduces a gating mechanism to dynamically balance semantic and geometric features.

#### Mechanism

- Learns adaptive weights between geometry and semantic branches
- Prevents over-reliance on a single modality
- Enables geometry-constrained cross-modal fusion

#### Advantages

- Improves semantic coherence
- Enhances spatial continuity
- Strengthens generalization and stability

> ⚙️ CGF improves inter-modal cooperation, making GazeLoom accurate and reliable in real-world in-cabin scenarios.

---

## 🧠 Architecture Overview

<div align="center">
  <img src="https://github.com/user-attachments/assets/acd7deab-6800-4aa1-a8d5-e065729dd186" width="820" />
</div>

The GazeLoom architecture estimates 3D driver gaze points through the following pipeline:

1. **Camera Input**
2. **Face Landmark Extraction**
3. **Head Pose Estimation**
4. **Eye Gaze Vector Modeling**
5. **Multi-modal Geometry Guidance**
6. **Cross-modal Gating Fusion**
7. **3D Gaze Point Prediction**

---

## 📊 Datasets & Results

| Dataset | AUC ↑ | L2 ↓ | AP ↑ |
|---|:---:|:---:|:---:|
| **GazeFollow** | **0.967** | **0.079** | - |
| **VideoAttentionTarget** | **0.953** | **0.098** | **0.942** |

> GazeLoom achieves strong performance across multiple benchmarks while maintaining a lightweight architecture.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourname/GAZELOOM.git
cd GAZELOOM
