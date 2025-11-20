# CaVL-Doc: Contrastive Aligned Vision-Language Document Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.12345-b31b1b.svg)](https://arxiv.org/abs/2510.12345) [![Dataset](https://img.shields.io/badge/Dataset-LA--CDIP-green)](https://github.com/jpcosta90/LA-CDIP)
[![Dataset](https://img.shields.io/badge/Dataset-RVL--CDIP-blue)](https://www.cs.cmu.edu/~aharley/rvl-cdip/)

This repository implements **CaVL-Doc**, an architecture and finetuning pipeline designed to generate **Contrastive Aligned Vision-Language Document Embeddings**. The goal is to optimize a Large Vision-Language Model (LVLM) to produce high-quality, unified document representations for tasks requiring robust **similarity comparison** and **zero-shot document classification**.

The pipeline is built on the **Autoregressive Multimodal Pre-Training** paradigm (similar to InternVL3), where image patches are processed as tokens alongside text. Our core contribution lies in optimizing the output layer to learn a high-quality embedding space via **Supervised Contrastive Loss**.

---

## CaVL-Doc Architecture and Strategy

CaVL-Doc focuses exclusively on enhancing the **representation quality** of the LVLM's output feature vector for document retrieval tasks.

![Model Architecture](docs/assets/model_architecture.png)

The system utilizes a frozen, pre-trained LVLM (e.g., InternVL3) and fine-tunes it along two primary technical pathways, both leveraging **Contrastive Loss** for metric learning:

### 1. Direct Fine-Tuning (QLoRA)

This pathway uses **QLoRA** to perform low-rank updates directly on the LVLM's weights (including potentially the Vision Encoder layers). The training objective is a **Supervised Contrastive Loss**, which pulls the embeddings of similar document pairs closer while pushing dissimilar pairs farther apart in the latent space.

### 2. Metric Head Optimization (Teacher-Student Curriculum Learning)

This advanced pathway fine-tunes a separate **Projection Head** (Metric Head) attached to the frozen LVLM's output. This head learns the contrastive representation.

* **Teacher-Student Strategy:** A **Reinforcement Learning (RL) "Teacher" agent** is employed to dynamically select the most informative and difficult training samples (a curriculum) to train the **"Student" (the Projection Head)**.
* **Purpose:** This strategy enhances generalization by focusing the head on hard-negative mining and complex boundaries, leading to more robust document embeddings.

This repository provides the core modules, trainers, and scripts to execute both fine-tuning strategies.

---

## Repository Structure

The project is structured to function as a reusable Python package (`cavl_doc`) dedicated to model training and representation generation.

```
.
├── checkpoints/              # Stores fine-tuned adapters (QLoRA) and Projection Heads (Student/Teacher)
├── data/                     # Datasets (LA-CDIP, RVL-CDIP) and pair files for training/evaluation
├── analysis/                 # Stores CSV files related to error analysis (optional)
├── results/                  # Stores master result logs and generated plots
├── scripts/                  # Executable Python scripts for running finetuning and evaluation experiments
└── src/                      # Source code library for the CaVL-Doc Python package
    ├── cavl_doc/             # The 'cavl-doc' Python module
    │   ├── __init__.py       # Package definition
    │   ├── model.py          # Definition of CaVLDocModel and Projection Heads
    │   ├── finetuning/       # Trainers (LoRA, RL) and Supervised Contrastive Loss functions
    │   ├── data_loaders/     # Dataset and sampler classes for contrastive pairs
    │   └── utils/            # Utility classes and helpers
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jpcosta90/CaVL-Doc.git 
    cd CaVL-Doc
    ```

2.  **Set up a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Install the project in editable mode** (recommended to make the `cavl_doc` package available):
    ```bash
    pip install -e .
    ```

4.  **Download Datasets:**
    * Place the document image datasets (LA-CDIP, RVL-CDIP) and their respective pair files inside the `data/` directory.

---

## Usage & Experimental Workflow (Finetuning Focus)

This section details how to generate and evaluate the **CaVL-Doc** embeddings.

### 1. Generating Fine-Tuned Models 🧠

The primary goal is to generate adapter checkpoints (QLoRA) or Projection Head checkpoints (RL-Teacher/Student). Both methods rely on the standard **supervised contrastive loss**.

#### **1.1. Fine-Tuning with QLoRA (`run_fine_tuning_lora.py`)**

This script trains QLoRA adapters using supervised contrastive loss, targeting low-rank updates to the base LVLM.

```bash
python scripts/run_fine_tuning_lora.py \
    --pairs-csv "data/LA-CDIP/train_pairs.csv" \
    --base-image-dir "path/to/your/images/" \
    --training-sample-size 1000 \
    --epochs 10 \
    --num-vision-layers-to-train 1 # Train LoRA for the last vision block
```

#### **1.2. Metric Head Optimization (RL-based Curriculum) (`run_rl_finetuning_full.py`)**

This script implements the **Teacher-Student Curriculum Learning** strategy to train a Projection Head for metric learning.

```bash
python scripts/run_rl_finetuning_full.py \
    --model-name "InternVL3-2B" \
    --pairs-csv "data/LA-CDIP/train_pairs.csv" \
    --base-image-dir "path/to/your/images/" \
    --training-sample-size 2000 \
    --epochs 3 \
    --projection-output-dim 512 \
    --load-in-4bit 
```
* This saves the trained **Student Head** (the artifact for evaluation) and the **Professor Model** (Teacher agent) in the `checkpoints/` directory.

### 2. Evaluating Embeddings (`run_evaluation.py`)

Use the main evaluation script to test the performance of the generated checkpoints (models or heads) on zero-shot document comparison tasks (measured by EER).

#### **Example: Evaluating a Fine-Tuned Model/Head**
To evaluate a model that has been fine-tuned with **QLoRA** or a trained **Projection Head** (Student), use the `--checkpoint-path` argument:

```bash
python scripts/run_evaluation.py \
    --evaluation-method "embedding" \
    --model-name "InternVL3-2B" \
    --prompt "<image> describe this document" \
    --pairs-csv "data/LA-CDIP/validation_pairs.csv" \
    --base-image-dir "path/to/your/images/" \
    --checkpoint-path "checkpoints/your_adapter_folder_name" \
    --metric "euclidean" \
    --plot
```
* `--checkpoint-path`: Points to the folder containing the trained QLoRA adapters or the trained Student Head (`student_head_epoch_*.pt`).

---

## Results 📊

*The table below focuses exclusively on the performance improvements achieved through fine-tuning, as generated by the `scripts/update_readme.py` utility.*

### LA-CDIP Results

| Method | EER (%) | Model/Adapter | Metric | Link Figura |
|:---|---:|:---|:---|:---|
| LA-CDIP_InternVL3-2B_Full_Head_2000-d3b0af_euclidean | 2.31 | LA-CDIP_InternVL3-2B_RL_Full_Head_2000_20251108-194952 | euclidean | [Link](results/plots/LA-CDIP_LA-CDIP_InternVL3-2B_Full_Head_2000-d3b0af_euclidean.png) |
| LA-CDIP_InternVL3-2B_Full_Head_8000-aafe46_euclidean | 2.51 | LA-CDIP_InternVL3-2B_RL_Full_Head_8000_20251111-005720 | euclidean | [Link](results/plots/LA-CDIP_LA-CDIP_InternVL3-2B_Full_Head_8000-aafe46_euclidean.png) |
| InternVL3-2B_prompt-d3b0af_euclidean | 2.99 | InternVL3-2B | euclidean | [Link](results/plots/LA-CDIP_InternVL3-2B_prompt-d3b0af_euclidean.png) |
| pixel_euclidean_baseline | 9.07 | N/A | euclidean | [Link](results/plots/LA-CDIP_pixel_euclidean_baseline.png) |

### RVL-CDIP Results

| Method | EER (%) | Model/Adapter | Metric | Link Figura |
|:---|---:|:---|:---|:---|
| InternVL3-2B_prompt-d3b0af_euclidean | 30.45 | InternVL3-2B | euclidean | [Link](results/plots/RVL-CDIP_InternVL3-2B_prompt-d3b0af_euclidean.png) |
| pixel_cosine_baseline | 36.3 | N/A | cosine | [Link](results/plots/RVL-CDIP_pixel_cosine_baseline.png) |
| pixel_euclidean_baseline | 44.8 | N/A | euclidean | [Link](results/plots/RVL-CDIP_pixel_euclidean_baseline.png) |

---

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{Costa2025CaVLDOC,
  title   = {CaVL-Doc: Contrastive Aligned Vision-Language Document Embeddings for Zero-Shot Classification},
  author  = {Joao Paulo Vieira Costa and Co-authors},
  journal = {Journal or Conference Name},
  year    = {2025},
  volume  = {XX},
  pages   = {XX--XX}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.