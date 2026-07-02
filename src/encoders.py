from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_dim: int, forward_kind: str = "plain"):
        super().__init__()
        self.model = model
        self.output_dim = output_dim
        self.forward_kind = forward_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.forward_kind == "hf_pixel_values":
            out = self.model(pixel_values=x)
            feat = extract_hf_features(out)
        elif self.forward_kind == "medimageinsight":
            feat = forward_medimageinsight(self.model, x)
        else:
            feat = self.model(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return F.normalize(feat.float(), dim=1)


def extract_hf_features(out: Any) -> torch.Tensor:
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state[:, 0]
    if isinstance(out, (tuple, list)):
        first = out[0]
        if first.ndim == 3:
            return first[:, 0]
        return first
    if torch.is_tensor(out):
        return out
    raise RuntimeError("Could not extract features from Hugging Face output")


def forward_medimageinsight(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):
        return model.encode_image(x)
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=x)
    out = model(pixel_values=x) if callable(model) else None
    return extract_hf_features(out)


def _load_checkpoint_if_available(model: nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model", "encoder", "backbone"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "").replace("backbone.", "").replace("encoder.", "")
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)


def build_torchvision_resnet(name: str = "resnet50", pretrained: bool = True, checkpoint_path: Optional[str] = None) -> EncoderWrapper:
    from torchvision import models
    weights = None
    if pretrained and name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = getattr(models, name)(weights=weights)
    dim = model.fc.in_features
    model.fc = nn.Identity()
    _load_checkpoint_if_available(model, checkpoint_path)
    return EncoderWrapper(model, dim)


def build_huggingface_vision(model_id: str, cache_dir: Optional[str] = None, trust_remote_code: bool = True) -> EncoderWrapper:
    from transformers import AutoModel, CLIPVisionModel
    try:
        model = CLIPVisionModel.from_pretrained(model_id, cache_dir=cache_dir)
    except Exception:
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    dim = getattr(getattr(model, "config", None), "hidden_size", None) or getattr(getattr(model, "config", None), "projection_dim", None) or 768
    return EncoderWrapper(model, int(dim), "hf_pixel_values")


def build_timm_encoder(model_id: str, pretrained: bool = True, checkpoint_path: Optional[str] = None) -> EncoderWrapper:
    import timm
    model = timm.create_model(model_id, pretrained=pretrained, num_classes=0)
    dim = int(getattr(model, "num_features", 0) or getattr(model, "embed_dim", 0) or 768)
    _load_checkpoint_if_available(model, checkpoint_path)
    return EncoderWrapper(model, dim)


def build_medimageinsight(model_id: str, cache_dir: Optional[str] = None) -> EncoderWrapper:
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        dim = int(getattr(getattr(model, "config", None), "hidden_size", 1024))
        return EncoderWrapper(model, dim, "medimageinsight")
    except Exception as exc:
        raise ImportError("MedImageInsight could not be loaded. Install its required package or provide a compatible Hugging Face cache.") from exc


def build_encoder_from_config(cfg: Dict[str, Any]) -> EncoderWrapper:
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})
    family = model_cfg.get("family") or model_cfg.get("backbone") or "torchvision"
    model_id = model_cfg.get("architecture_or_model_id") or model_cfg.get("hf_repo_id") or model_cfg.get("tag") or "resnet50"
    checkpoint = model_cfg.get("checkpoint_path") or model_cfg.get("pretrained_checkpoint")
    cache_dir = paths_cfg.get("model_cache_dir") or model_cfg.get("cache_dir")
    pretrained = bool(model_cfg.get("pretrained", True))
    if family == "torchvision":
        return build_torchvision_resnet(model_id, pretrained, checkpoint)
    if family == "self_supervised_resnet50":
        return build_torchvision_resnet("resnet50", pretrained, checkpoint)
    if family in {"huggingface_clip", "huggingface_vision_transformer"}:
        return build_huggingface_vision(model_id, cache_dir)
    if family == "timm_vision_transformer":
        return build_timm_encoder(model_id, pretrained, checkpoint)
    if family == "medimageinsight" or str(model_id).lower().endswith("medimageinsights"):
        return build_medimageinsight(model_id, cache_dir)
    raise ValueError(f"Unsupported encoder family {family}")
