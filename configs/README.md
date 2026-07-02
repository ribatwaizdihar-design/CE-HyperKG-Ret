# Configs

This folder contains YAML configuration files for CE-HyperKG-Ret experiments.

## Structure

```text
configs/
├── baselines/
├── proposed/
├── ablations/
├── evaluation/
└── runtime/
```

## Baselines

The `baselines/` folder contains configurations for:

1. ResNet-50
2. SimCLR
3. MoCo v2
4. CLIP
5. PubMedCLIP
6. DINOv2
7. RAD-DINO
8. MedImageInsight

## Proposed Model

The `proposed/` folder contains the final CE-HyperKG-Ret V6 configuration using seeds 42, 123, and 2025.

## Ablations

The `ablations/` folder contains V1 to V6 experiment configurations.

## Evaluation

The `evaluation/` folder contains the retrieval evaluation protocol.

## Runtime

The `runtime/` folder contains environment and runtime settings.

## Example Commands

Run one baseline:

```bash
python -m src.train_baseline --config configs/baselines/01_resnet50_baseline.yaml
```

Run the final model:

```bash
python -m src.train_proposed --config configs/proposed/09_ce_hyperkg_ret_proposed_v6_all_seeds.yaml
```
