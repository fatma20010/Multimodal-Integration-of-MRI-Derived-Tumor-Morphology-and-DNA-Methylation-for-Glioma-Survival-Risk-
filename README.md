# Multimodal Integration of MRI-Derived Tumor Morphology and DNA Methylation for Glioma Survival Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Research-Glioma%20Survival%20Analysis-purple.svg)]()
[![Models](https://img.shields.io/badge/Models-U--Net%20%7C%20V--Net%20%7C%20SegNet-green.svg)]()

## Overview

This repository contains the implementation of a multimodal deep-learning framework for glioma survival risk assessment, integrating MRI-derived tumor morphology with DNA methylation information.

The framework investigates how complementary imaging and molecular characteristics can contribute to patient-level survival risk stratification. It combines 3D brain tumor segmentation, quantitative morphological feature extraction, methylation-based risk modelling, and multimodal learning strategies.

The repository provides the implementation, experimental workflows, and trained segmentation model checkpoints associated with the research.

## Framework

The proposed pipeline consists of the following main stages:

1. **3D MRI tumor segmentation:** Evaluation and comparison of U-Net, V-Net, and SegNet architectures for volumetric tumor segmentation.
2. **Morphological feature extraction:** Derivation of 14 quantitative descriptors from segmented tumor regions to characterize tumor morphology.
3. **Methylation-based modelling:** Processing of DNA methylation features and development of molecular risk scores using survival modelling.
4. **Genomic score reconstruction:** Investigation of the recovery of molecular risk scores when only partial methylation information is available.
5. **Multimodal integration:** Evaluation of score-level fusion, score-based Vision Transformers, and feature-based Vision Transformer approaches.
6. **Survival risk assessment:** Evaluation of the predictive contribution of imaging, molecular, and integrated representations using the concordance index (C-index).

## Model Architectures

### 3D Tumor Segmentation

Three segmentation architectures are included:

| Architecture | Description |
|---|---|
| U-Net | Encoder–decoder architecture with skip connections for volumetric tumor segmentation |
| V-Net | Volumetric convolutional architecture designed for 3D medical image segmentation |
| SegNet | Encoder–decoder segmentation architecture with pooling-index-based upsampling |

### Multimodal Survival Modelling

The survival analysis pipeline investigates several configurations:

- MRI-derived morphological features.
- DNA methylation-based survival risk modelling.
- Cox-based score-level integration.
- Score-based Vision Transformer fusion.
- Feature-based Vision Transformer fusion.

## Trained Model Checkpoints

<ins>Pretrained segmentation weights are distributed through GitHub Releases rather than being committed to the repository.</ins>

**Release:** [v1.0.0 – Initial Multimodal Glioma Segmentation Models](../../releases/tag/v1.0.0)

The release includes the following trained checkpoints:

| Model | Checkpoint |
|---|---|
| U-Net | Trained 3D U-Net weights (`.pth`) |
| V-Net | Trained 3D V-Net weights (`.pth`) |
| SegNet | Trained 3D SegNet weights (`.pth`) |

Download the required checkpoint from the [Releases page](../../releases).

After downloading, place the checkpoint in the corresponding model directory or update the checkpoint path in the inference configuration.

> **Note:** The checkpoints are model weights, not standalone executable applications. The corresponding model architecture and inference implementation are provided in the repository.

## Experimental Results

Selected results obtained in the research experiments include:

| Experiment | Result |
|---|---:|
| 3D U-Net full-volume mean Dice | 0.8972 |
| MRI-only survival prediction (GBM) | 0.6169 C-index |
| Methylation-only survival prediction (GBM) | 0.5673 C-index |
| Cox score-level fusion | 0.6154 C-index |
| Score-based Vision Transformer | 0.6332 C-index |
| Feature-based Vision Transformer (K = 1000) | **0.6502 C-index** |

These results correspond to the reported experimental configurations and should be interpreted in the context of their respective cohorts and evaluation protocols.

## Repository Structure

The repository is organized around the main stages of the research workflow:

```text
.
├── Segmentation/
│   └── Step1-Segmentation/
│       ├── U-Net/
│       ├── V-Net/
│       └── SegNet/
│
├── Survival-Analysis/
│
├── Genomic-Analysis/
│
├── Multimodal-Fusion/
│
├── requirements.txt
├── README.md
└── ...
```

The directory names above illustrate the intended organization. Refer to the actual repository folders for the current implementation and scripts.

## Installation

Clone the repository:

```bash
git clone https://github.com/fatma20010/Multimodal-Integration-of-MRI-Derived-Tumor-Morphology-and-DNA-Methylation-for-Glioma-Survival-Risk-.git

cd Multimodal-Integration-of-MRI-Derived-Tumor-Morphology-and-DNA-Methylation-for-Glioma-Survival-Risk-
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU-enabled experiments, install a PyTorch build compatible with your CUDA environment, following the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Using the Trained Models

1. Open the repository's [Releases page](../../releases).
2. Download the desired U-Net, V-Net, or SegNet checkpoint.
3. Place the downloaded `.pth` file in the appropriate checkpoint directory.
4. Load the corresponding model architecture from the implementation.
5. Configure the input MRI volumes and preprocessing parameters.
6. Run the relevant inference or evaluation script.

The exact model configuration, preprocessing, and inference settings should match those used during training.

## Datasets and Data Availability

The research uses brain MRI data and DNA methylation information for glioma analysis.

Dataset access, availability, and usage conditions are subject to the policies of the respective data providers. This repository does not imply redistribution rights for third-party clinical, imaging, or molecular datasets.

Researchers should obtain the necessary permissions and follow the applicable data-use agreements before accessing or reusing patient-level data.

## Reproducibility

For reproducibility, experiments should use the corresponding model architecture, preprocessing pipeline, checkpoint, and evaluation protocol.

Please consult the implementation and configuration files for experiment-specific parameters.

## Citation

If you use this repository in your research, please cite the associated publication once it is published.

A citation entry can be added here:

```bibtex
@article{hammedi_glioma_multimodal,
  title   = {Multimodal Integration of MRI-Derived Tumor Morphology
             and DNA Methylation for Glioma Survival Risk Assessment},
  author  = {Hammedi, Fatma and collaborators},
  journal = {To be updated},
  year    = {2026}
}
```

## Acknowledgments

This research was conducted in the context of academic research in artificial intelligence and medical image analysis, with a focus on multimodal glioma characterization and survival risk assessment.

## License

License information will be specified separately. Please contact the repository authors regarding reuse until a license has been added.

## Contact

**Fatma Hammedi**  
GitHub: [@fatma20010](https://github.com/fatma20010)
