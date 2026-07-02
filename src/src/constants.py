DEFAULT_SEEDS = (42, 123, 2025)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
FUSION_RULE = (
    (0.40, 0.70, 0.20, 0.10),
    (0.70, 0.50, 0.30, 0.20),
    (1.01, 0.30, 0.35, 0.35),
)
FINAL_V6 = {
    "backbone": "MedImageInsight",
    "input_resolution": 512,
    "min_cui_freq": 5,
    "retained_vocabulary_size": 1916,
    "top_n_visual": 200,
    "top_l_predicted_cuis": 10,
    "cui_threshold": 0.30,
    "lambda_cui": 1.00,
    "lambda_kg": 0.10,
    "lambda_counterfactual": 0.10,
    "margin": 0.20,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 25,
    "early_stopping_patience": 5,
    "validation_metric": "generic_filtered_ndcg_at_5",
    "mixed_precision": True,
    "seeds": DEFAULT_SEEDS,
    "bootstrap_resamples": 10000,
}
BASELINES = {
    "resnet50": {
        "display_name": "ResNet-50",
        "family": "torchvision",
        "architecture_or_model_id": "resnet50",
        "image_size": 224,
    },
    "simclr": {
        "display_name": "SimCLR",
        "family": "self_supervised_resnet50",
        "architecture_or_model_id": "resnet50",
        "image_size": 224,
    },
    "moco_v2": {
        "display_name": "MoCo v2",
        "family": "self_supervised_resnet50",
        "architecture_or_model_id": "resnet50",
        "image_size": 224,
    },
    "clip": {
        "display_name": "CLIP",
        "family": "huggingface_clip",
        "architecture_or_model_id": "openai/clip-vit-base-patch32",
        "image_size": 224,
    },
    "pubmedclip": {
        "display_name": "PubMedCLIP",
        "family": "huggingface_clip",
        "architecture_or_model_id": "flaviagiammarino/pubmed-clip-vit-base-patch32",
        "image_size": 224,
    },
    "dinov2": {
        "display_name": "DINOv2",
        "family": "timm_vision_transformer",
        "architecture_or_model_id": "vit_small_patch14_reg4_dinov2.lvd142m",
        "image_size": 224,
    },
    "rad_dino": {
        "display_name": "RAD-DINO",
        "family": "huggingface_vision_transformer",
        "architecture_or_model_id": "microsoft/rad-dino",
        "image_size": 518,
    },
    "medimageinsight": {
        "display_name": "MedImageInsight",
        "family": "medimageinsight",
        "architecture_or_model_id": "lion-ai/MedImageInsights",
        "image_size": 512,
    },
}
METRICS_AT_K = ("cui", "precision", "recall", "map", "mrr", "ndcg", "mean_jaccard")
