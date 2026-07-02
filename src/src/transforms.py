from typing import Sequence


def build_image_transform(image_size: int | Sequence[int], normalize: str = "imagenet", train: bool = False):
    from torchvision import transforms
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = tuple(image_size)
    ops = []
    if train:
        ops.extend([transforms.Resize(size), transforms.RandomHorizontalFlip(p=0.5)])
    else:
        ops.append(transforms.Resize(size))
    ops.append(transforms.ToTensor())
    if normalize == "imagenet":
        ops.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    elif normalize == "clip":
        ops.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization {normalize}")
    return transforms.Compose(ops)


def normalization_for_family(family: str) -> str:
    if family in {"huggingface_clip"}:
        return "clip"
    return "imagenet"
