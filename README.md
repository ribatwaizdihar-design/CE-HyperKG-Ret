# CE-HyperKG-Ret

Official implementation of **CE-HyperKG-Ret** for medical image retrieval on ROCOv2.

This repository contains implementations of eight baseline image retrieval models and the proposed CE-HyperKG-Ret model.

## Repository Structure

```text
CE-HyperKG-Ret/
├── configs/
│   ├── baselines/
│   ├── proposed/
│   ├── ablations/
│   ├── evaluation/
│   └── runtime/
├── notebooks/
├── src/
├── data/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Models

The following baseline encoders are included:

1. ResNet-50
2. SimCLR
3. MoCo v2
4. CLIP
5. PubMedCLIP
6. DINOv2
7. RAD-DINO
8. MedImageInsight

The proposed model is:

```text
CE-HyperKG-Ret V6
```

## Proposed Model Settings

The final CE-HyperKG-Ret V6 setting follows:

```text
Backbone: MedImageInsight
Input size: 512 × 512
Minimum CUI frequency: 5
CUI vocabulary size: 1916
Top visual candidates: 200
Top predicted CUIs: 10
CUI threshold: 0.30
Batch size: 32
Maximum epochs: 25
Early stopping patience: 5
Learning rate: 1e-4
Weight decay: 1e-4
Lambda 1: 1.00
Lambda 2: 0.10
Lambda 3: 0.10
Margin: 0.20
Seeds: 42, 123, 2025
Bootstrap resamples: 10000
```

## Installation

Create a Python environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is not included in this repository.

Place ROCOv2 or your processed dataset under:

```text
data/
```

Recommended structure:

```text
data/
├── train/
├── validation/
├── test/
├── metadata/
└── cui_annotations/
```

Set the dataset path before running:

```bash
export DATA_ROOT=/path/to/roco_v2
```

On Windows PowerShell:

```powershell
$env:DATA_ROOT="C:\path\to\roco_v2"
```

## Running Baselines

Run one baseline:

```bash
python -m src.train_baseline --config configs/baselines/01_resnet50_baseline.yaml
```

Run all baselines:

```bash
python -m src.run_all_baselines --config-dir configs/baselines --output results/all_baselines_summary.csv
```

## Running the Proposed Model

Run the final proposed CE-HyperKG-Ret V6 model:

```bash
python -m src.train_proposed --config configs/proposed/09_ce_hyperkg_ret_proposed_v6_all_seeds.yaml
```

## Running Ablations

Example:

```bash
python -m src.train_proposed --config configs/ablations/v6_ce_hyperkg_ret_ablation.yaml
```

## Evaluation Metrics

The repository supports the following evaluation metrics:

```text
CUI@K
Precision@K
Recall@K
mAP@K
MRR@K
NDCG@K
MeanJaccard@K
```

## Notebooks

The `notebooks/` folder contains:

```text
01_resnet50_baseline.ipynb
02_simclr_baseline.ipynb
03_moco_v2_baseline.ipynb
04_clip_baseline.ipynb
05_pubmedclip_baseline.ipynb
06_dinov2_baseline.ipynb
07_rad_dino_baseline.ipynb
08_medimageinsight_baseline.ipynb
09_ce_hyperkg_ret_proposed_all_seeds.ipynb
```



## Reproducibility

The proposed model uses the following seeds:

```text
42, 123, 2025
```

Use the configs in `configs/` to reproduce the experiments.

## Citation

If you use this code, please cite the related publication.

```bibtex
@article{cehyperkgret,
  title={CE-HyperKG-Ret},
  author={},
  journal={},
  year={}
}
```

## License

This project is released under the MIT License.
