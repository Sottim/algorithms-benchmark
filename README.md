# Benchmarking of DL Algorithms

## Project Overview

This repository provides a unified framework for benchmarking state-of-the-art deep learning algorithms for cell segmentation and classification in digital pathology. It includes implementations and evaluation pipelines for three major models:

- **CellViT**: Vision Transformers for precise cell segmentation and classification. The exiting pipeline was modified to include the patch level input apart from WSI input.
- **CellVTA**: Vision Transformer Adapter for enhanced cell segmentation and classification.
- **HoVer-Net**: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.

Each model is organized in its own subdirectory with dedicated configuration, environment, and documentation files.

## Repository Structure

```
models-evaluation/
├── models/
│   ├── cellvit/      # CellViT implementation
│   ├── cellvta/      # CellVTA implementation
│   └── hovernet/     # HoVer-Net implementation
├── notebooks/        # Example and analysis notebooks
│   ├── cellvit/
│   ├── hovernet/
│   └── input_dir/    # Example input data (patches, masks)
```

## Installation & Setup

Each model has its own environment and requirements files. Please refer to the respective subproject README for detailed setup instructions. In general:

1. **Clone the repository**
2. **Set up the environment for the desired model** (example for CellViT):
   ```bash
   cd models/cellvit/CellViT
   conda env create -f environment.yml
   conda activate cellvit_env
   ```
3. **Download pretrained weights and datasets** as described in the subproject README.

## Usage

- **CellViT**: See `models/cellvit/CellViT/README.md` for training and inference commands.
- **CellVTA**: See `models/cellvta/CellVTA/README.md` for setup, training, and inference.
- **HoVer-Net**: See `models/hovernet/hover_net/README.md` for usage details.

## Notebooks

Example and analysis notebooks are provided in the `notebooks/` directory for each model. These include:
- Patch extraction
- Prediction result analysis
- Overlaying ground truth
- Result visualization

## References
- [CellViT: Vision Transformers for Precise Cell Segmentation and Classification](https://github.com/TIO-IKIM/CellViT)
- [CellVTA: Vision Transformer Adapter for Dense Predictions](https://github.com/TIO-IKIM/CellVTA)
- [HoVer-Net: Simultaneous Segmentation and Classification of Nuclei](https://github.com/vqdang/hover_net)

For detailed citations and acknowledgements, see the respective subproject READMEs. 