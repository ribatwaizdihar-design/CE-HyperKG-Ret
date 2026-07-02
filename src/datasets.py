from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .constants import IMAGE_EXTENSIONS
from .cui import normalize_cui_set, parse_cuis_from_value

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class RocoRecord:
    image_id: str
    split: str
    image_path: str
    cuis: List[str]
    caption: str = ""


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _lower_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).lower().strip(): c for c in df.columns}


def infer_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lm = _lower_map(df)
    for name in candidates:
        key = name.lower().strip()
        if key in lm:
            return lm[key]
    for col in df.columns:
        low = str(col).lower()
        if any(name.lower() in low for name in candidates):
            return col
    if required:
        raise ValueError(f"Could not infer required column from {list(df.columns)}")
    return None


def infer_id_column(df: pd.DataFrame) -> str:
    return infer_column(df, ["image_id", "imageid", "id", "roco_id", "name", "filename", "file_name", "image", "path"])


def infer_caption_column(df: pd.DataFrame) -> Optional[str]:
    return infer_column(df, ["caption", "text", "sentence", "report", "description"], required=False)


def infer_cui_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        low = str(col).lower()
        if any(x in low for x in ["cui", "concept", "umls"]):
            cols.append(col)
    return cols


def infer_path_column(df: pd.DataFrame) -> Optional[str]:
    return infer_column(df, ["image_path", "filepath", "file_path", "path", "filename", "file_name"], required=False)


def normalize_image_id(value: Any) -> str:
    s = str(value).strip()
    return Path(s).stem if any(s.lower().endswith(ext) for ext in IMAGE_EXTENSIONS) else s


def candidate_table_paths(data_root: str | Path, split: str, kind: str) -> List[Path]:
    data_root = Path(data_root)
    names = []
    if kind == "caption":
        names = [f"{split}_captions.csv", f"{split}.csv", f"{split}_metadata.csv", f"{split}_manifest.csv", f"{split}_captions.tsv"]
    if kind == "concept":
        names = [f"{split}_concepts.csv", f"{split}_concepts_manual.csv", f"{split}_cuis.csv", f"{split}_concepts.tsv", f"{split}_manual_concepts.csv"]
    out = []
    for name in names:
        out.extend([data_root / name, data_root / split / name])
    return out


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_image_path(data_root: str | Path, split: str, image_id: str, raw_path: Optional[Any] = None, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> Optional[str]:
    data_root = Path(data_root)
    if raw_path is not None and str(raw_path).strip() and str(raw_path).lower() != "nan":
        p = Path(str(raw_path))
        candidates = [p, data_root / p, data_root / split / p]
        for cand in candidates:
            if cand.exists() and cand.is_file():
                return str(cand)
    stem = normalize_image_id(image_id)
    folders = [data_root, data_root / split, data_root / "images", data_root / split / "images", data_root / "image", data_root / split / "image"]
    for folder in folders:
        for ext in extensions:
            cand = folder / f"{stem}{ext}"
            if cand.exists():
                return str(cand)
    return None


def aggregate_concepts(df: pd.DataFrame) -> Dict[str, List[str]]:
    id_col = infer_id_column(df)
    cui_cols = infer_cui_columns(df)
    if not cui_cols:
        return {}
    groups: Dict[str, List[Any]] = {}
    for _, row in df.iterrows():
        image_id = normalize_image_id(row[id_col])
        groups.setdefault(image_id, [])
        for col in cui_cols:
            groups[image_id].append(row[col])
    return {k: normalize_cui_set(v) for k, v in groups.items()}


def load_split_records(data_root: str | Path, split: str, caption_file: Optional[str] = None, concept_file: Optional[str] = None, image_extensions: Sequence[str] = IMAGE_EXTENSIONS, require_image: bool = True, limit: Optional[int] = None) -> List[RocoRecord]:
    data_root = Path(data_root)
    caption_path = Path(caption_file) if caption_file else first_existing(candidate_table_paths(data_root, split, "caption"))
    concept_path = Path(concept_file) if concept_file else first_existing(candidate_table_paths(data_root, split, "concept"))
    if caption_path is None and concept_path is None:
        raise FileNotFoundError(f"No manifest file found for split {split} under {data_root}")
    base_df = read_table(caption_path) if caption_path is not None else read_table(concept_path)
    id_col = infer_id_column(base_df)
    cap_col = infer_caption_column(base_df)
    path_col = infer_path_column(base_df)
    direct_cui_cols = infer_cui_columns(base_df)
    concept_map = aggregate_concepts(read_table(concept_path)) if concept_path is not None else {}
    records: List[RocoRecord] = []
    seen = set()
    for _, row in base_df.iterrows():
        image_id = normalize_image_id(row[id_col])
        if image_id in seen:
            continue
        seen.add(image_id)
        raw_path = row[path_col] if path_col else None
        image_path = resolve_image_path(data_root, split, image_id, raw_path, image_extensions)
        if require_image and image_path is None:
            continue
        caption = str(row[cap_col]) if cap_col and not pd.isna(row[cap_col]) else ""
        row_cuis = []
        for col in direct_cui_cols:
            row_cuis.extend(parse_cuis_from_value(row[col]))
        cuis = normalize_cui_set([row_cuis, concept_map.get(image_id, [])])
        records.append(RocoRecord(image_id=image_id, split=split, image_path=image_path or "", cuis=cuis, caption=caption))
        if limit is not None and len(records) >= limit:
            break
    return records


def records_to_dataframe(records: Sequence[RocoRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {"image_id": r.image_id, "split": r.split, "image_path": r.image_path, "cuis": " ".join(r.cuis), "caption": r.caption}
        for r in records
    ])


class RocoImageDataset(Dataset):
    def __init__(self, records: Sequence[RocoRecord], labels: Optional[Any] = None, transform: Optional[Any] = None):
        self.records = list(records)
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        item = {"image": image, "image_id": record.image_id, "cuis": record.cuis, "image_path": record.image_path}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item
