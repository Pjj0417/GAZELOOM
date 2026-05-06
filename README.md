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

The preprocessing scripts convert raw annotations into a unified JSON format for each split, including **head bounding boxes**, **gaze points**, **in/out labels**, and metadata required by GazeLoom.

<div align="center">

| Dataset | Download | Preprocessing Script |
|:---:|:---:|:---:|
| 👀 **GazeFollow** | [Download](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset) | `data_prep/preprocess_gazefollow.py` |
| 🎥 **VideoAttentionTarget** | [Download](https://github.com/ejcgt/attention-target-detection?tab=readme-ov-file#dataset-1) | `data_prep/preprocess_vat.py` |
| 🧒 **ChildPlay** | [Download](https://www.idiap.ch/en/scientific-research/data/childplay-gaze) | `data_prep/preprocess_childplay.py` |
| 🛒 **GOO-Real** | [Download](https://github.com/upeee/GOO-GAZE2021/blob/main/dataset/gooreal-download.txt) | `data_prep/preprocess_goo_real.py` |

</div>

---

### 👀 GazeFollow

```bash
python data_prep/preprocess_gazefollow.py \
  --data_path /path/to/gazefollow/data_new
```

### 🎥 VideoAttentionTarget

```bash
python data_prep/preprocess_vat.py \
  --data_path /path/to/videoattentiontarget
```

### 🧒 ChildPlay

```bash
python data_prep/preprocess_childplay.py \
  --data_path /path/to/childplay
```

### 🛒 GOO-Real

```bash
python data_prep/preprocess_goo_real.py \
  --data_path /path/to/goo_real
```

---

After preprocessing, each dataset directory will contain JSON annotation files that can be directly used for **GazeLoom training and evaluation**.
## 🌊 Depth Map Extraction

GazeLoom uses **Depth Anything V2** to generate monocular depth maps as geometric guidance.

### Pretrained Weights

Download official Depth Anything V2 checkpoints:

| Model | Checkpoint |
|---|---|
| Depth-Anything-V2-Small | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) |
| Depth-Anything-V2-Base | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base) |
| Depth-Anything-V2-Large | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) |

Place the downloaded checkpoint under:

```text
checkpoints/
└── depth_anything_v2_vitl.pth
```

### Extract Depth Maps

```bash
python depthany/depth.py \
  --img_path /path/to/images \
  --outdir /path/to/depth \
  --encoder vitl \
  --checkpoint checkpoints/depth_anything_v2_vitl.pth \
  --input_size 518 \
  --grayscale
```

For GazeLoom datasets, the generated depth maps should preserve the same relative image paths:

```text
dataset_root/
├── gazefollow/
│   └── xxx.jpg
├── videoattentiontarget/
│   └── xxx.jpg
└── depth/
    └── xxx.png
```
## 🏋️ Train

We provide training scripts in `scripts/` for training GazeLoom with the **SimDINOv2** backbone.

Before running the training script, please:

- 📥 Download the dataset and run the preprocessing script following the [Data Processing](#data-processing) section.
- 🌊 Prepare the extracted depth maps if geometry-guided training is enabled.
- 📊 Optionally install `wandb` for metric logging:

```bash
pip install wandb
```

By default, checkpoints are saved to `./experiments`.  
You can use `--ckpt_save_dir` to customize the checkpoint directory.

---

### 👀 GazeFollow

Train GazeLoom with the **SimDINOv2 ViT-B/14** backbone:

```bash
python scripts/train_gazefollow.py \
  --data_path /path/to/gazefollow/data_new \
  --model gazeloom_cgf_simdinov2_vitb14_inout \
  --exp_name train_gazeloom_simdinov2_vitb14_gazefollow \
  --batch_size 48 \
  --max_epochs 30 \
  --lr 5e-4
```

Train GazeLoom with the **SimDINOv2 ViT-L/14** backbone:

```bash
python scripts/train_gazefollow.py \
  --data_path /path/to/gazefollow/data_new \
  --model gazeloom_cgf_simdinov2_vitl14_inout \
  --exp_name train_gazeloom_simdinov2_vitl14_gazefollow \
  --batch_size 32 \
  --max_epochs 30 \
  --lr 5e-4
```

---

### 🔁 Resume Training

Resume training from a saved checkpoint:

```bash
python scripts/train_gazefollow.py \
  --data_path /path/to/gazefollow/data_new \
  --model gazeloom_cgf_simdinov2_vitb14_inout \
  --resume /path/to/checkpoint.pt \
  --exp_name resume_gazeloom_simdinov2_vitb14
```

---

### 🔓 Backbone Fine-tuning

By default, the backbone is frozen.  
To fine-tune the last several SimDINOv2 transformer blocks, use `--unfreeze_layers`:

```bash
python scripts/train_gazefollow.py \
  --data_path /path/to/gazefollow/data_new \
  --model gazeloom_cgf_simdinov2_vitb14_inout \
  --unfreeze_layers 2 \
  --exp_name finetune_gazeloom_simdinov2_vitb14
```

---

### ⚙️ Main Arguments

| Argument | Description |
|---|---|
| `--model` | Model name, e.g. `gazeloom_cgf_simdinov2_vitb14_inout` |
| `--data_path` | Path to the preprocessed dataset |
| `--ckpt_save_dir` | Directory for saving checkpoints |
| `--exp_name` | Experiment name |
| `--batch_size` | Training batch size |
| `--max_epochs` | Number of training epochs |
| `--lr` | Learning rate for trainable heads |
| `--resume` | Path to checkpoint for resuming training |
| `--unfreeze_layers` | Number of final backbone layers to fine-tune |

## 🎨 Visualization

We provide a visualization script in `scripts/visualize.py` for qualitative analysis of GazeLoom predictions.

The script automatically detects faces using RetinaFace, predicts gaze heatmaps for each detected person, and saves the visualization results, including:

- 🟧 detected head bounding boxes
- 🔴 predicted gaze target points
- 🟢 high-response heatmap regions
- 📍 gaze direction lines
- 🌈 heatmap overlay images
- 📄 JSON prediction results

### Run Visualization

```bash
python scripts/visualize.py \
  --image_dir test \
  --depth_dir test_depth \
  --ckpt_path checkpoints/epoch_14.pt \
  --model_name gazeloom_cgf_simdinov2_vitb14_inout \
  --output_dir output
```

### Arguments

| Argument | Description |
|---|---|
| `--image_dir` | Directory containing input images |
| `--depth_dir` | Directory containing extracted depth maps |
| `--ckpt_path` | Path to the trained model checkpoint |
| `--model_name` | Model architecture used for inference |
| `--output_dir` | Directory for saving visualization results |
| `--inout_threshold` | Threshold for filtering out-of-frame gaze predictions |
| `--heatmap_threshold` | Threshold for highlighting high-response gaze regions |

### Output Files

After running the script, results will be saved under the output directory:

```text
output/
├── image_result.jpg
├── image_heatmap_0.jpg
├── image_overlay_0.jpg
└── gaze_predictions.json
```
## 🧪 Evaluation

We provide evaluation scripts in `scripts/` to validate GazeLoom on standard gaze-target benchmarks.

Before evaluation, please make sure that:

- 📥 The dataset has been downloaded.
- 🗂️ The preprocessing script has been executed.
- 🌊 Depth maps have been generated if the model uses geometry guidance.
- 📦 The pretrained checkpoint has been downloaded.

---

### 👀 GazeFollow

Evaluate GazeLoom on the GazeFollow test split:

```bash
python scripts/eval_gazefollow.py \
  --data_path /path/to/gazefollow/data_new \
  --model_name gazeloom_cgf_simdinov2_vitl14 \
  --ckpt_path /path/to/checkpoint.pt \
  --batch_size 128
```

The script reports:

| Metric | Description |
|---|---|
| **AUC ↑** | Area under the gaze heatmap ROC curve |
| **Avg L2 ↓** | Average L2 distance between prediction and ground-truth gaze points |
| **Min L2 ↓** | Minimum L2 distance to the closest ground-truth annotation |

---

### 🎥 VideoAttentionTarget

Evaluate GazeLoom on VideoAttentionTarget:

```bash
python scripts/eval_vat.py \
  --data_path /path/to/videoattentiontarget \
  --model_name gazeloom_cgf_simdinov2_vitl14_inout \
  --ckpt_path /path/to/checkpoint.pt \
  --batch_size 64
```

For VideoAttentionTarget, the `_inout` model is recommended because the dataset includes both in-frame and out-of-frame gaze targets.

---

### 🧒 ChildPlay

Evaluate GazeLoom on ChildPlay:

```bash
python scripts/eval_childplay.py \
  --data_path /path/to/childplay \
  --model_name gazeloom_cgf_simdinov2_vitl14_inout \
  --ckpt_path /path/to/checkpoint.pt \
  --batch_size 64
```

---

### 🛒 GOO-Real

Evaluate GazeLoom on GOO-Real:

```bash
python scripts/eval_goo_real.py \
  --data_path /path/to/goo_real \
  --model_name gazeloom_cgf_simdinov2_vitl14 \
  --ckpt_path /path/to/checkpoint.pt \
  --batch_size 64
```

---

### 📊 Example Output

```text
Running on cuda
Evaluating: 100%|████████████████████| 100/100
AUC: 0.967
Avg L2: 0.112
Min L2: 0.079
```

The generated visualizations can be used to inspect gaze direction, predicted attention targets, and model behavior under different driving scenarios.

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
  <img src="https://github.com/peng86584-commits/GAZELOOM/blob/main/fig1.png?raw=true" width="720" />
</div>

---

## 🎯 Stage 1 — Feature Extraction

A pre-trained self-supervised backbone, **SimDINOv2**, is used to encode driving scene images and generate robust global visual representations.

<div align="center">
  <img src="https://github.com/peng86584-commits/GAZELOOM/blob/main/fig2.png?raw=true" width="720" />
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
  <img src="https://github.com/peng86584-commits/GAZELOOM/blob/main/fig3.png?raw=true" width="720" />
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


git clone https://github.com/yourname/GAZELOOM.git
cd GAZELOOM



## 🔁 Reproducibility

To facilitate reproducibility, we provide the training seeds, hyperparameters, optimizer settings, and learning-rate schedules used in our experiments.

> ⚠️ Due to hardware differences, CUDA/cuDNN behavior, and dataloader randomness, exact numerical results may slightly vary across different environments.

### 🧪 Training Environment

| Item | Configuration |
|---|---|
| 🧠 Framework | PyTorch 2.2.0 |
| ⚙️ CUDA | CUDA 12.1 |
| 🖥️ GPU | NVIDIA GPU |
| 📦 Backbone | SimDINOv2 |
| 📐 Input Size | `448 × 448` |
| 🔥 Heatmap Size | `64 × 64` |

### 🎲 Random Seeds

All experiments are initialized with fixed random seeds:

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

### ⚙️ Default Training Configuration

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Batch Size | 48 |
| Max Epochs | 30 |
| Base Learning Rate | `5e-4` |
| Backbone Learning Rate | `1e-5` |
| Scheduler | CosineAnnealingLR |
| Minimum LR | `1e-7` |
| Weight Decay | Not used by default |
| Backbone | Frozen by default |
| Drop Path | `0.1` |
| CGF Groups | `8` |

### 📊 Dataset-specific Schedule

| Dataset | Initialization | Epochs | Model |
|---|---|---:|---|
| 👀 GazeFollow | From SimDINOv2 backbone | 30 | `gazeloom_cgf_simdinov2_vitb14_inout` |
| 🎥 VideoAttentionTarget | Fine-tuned from GazeFollow checkpoint | 8 | `gazeloom_cgf_simdinov2_vitb14_inout` |
| 🧒 ChildPlay | Fine-tuned from GazeFollow checkpoint | 3 | `gazeloom_cgf_simdinov2_vitb14_inout` |
| 🛒 GOO-Real | Fine-tuned from GazeFollow checkpoint | 3 | `gazeloom_cgf_simdinov2_vitb14` |

### 🔓 Backbone Fine-tuning

By default, the SimDINOv2 backbone is frozen.  
If needed, the last `N` Transformer blocks can be unfrozen using:

```bash
--unfreeze_layers N
```

For example:

```bash
python scripts/train_gazefollow.py \
  --data_path /path/to/gazefollow/data_new \
  --model gazeloom_cgf_simdinov2_vitb14_inout \
  --batch_size 48 \
  --max_epochs 30 \
  --lr 5e-4 \
  --unfreeze_layers 0
```

### 📝 Notes

- The reported results are obtained using the configuration above.
- Checkpoints are saved after each epoch under `./experiments`.
- Training can be resumed with `--resume /path/to/checkpoint.pt`.
- Small performance variations may occur due to GPU type, CUDA kernels, and multi-worker dataloading.

