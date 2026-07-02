import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import yaml
except Exception as exc:
    raise ImportError("pyyaml is required to load YAML configs") from exc

_PATTERN = re.compile(r"^\$\{([^:}]+)(?::([^}]*))?\}$")


def _expand_string(value: str) -> str:
    match = _PATTERN.match(value)
    if match:
        key, default = match.group(1), match.group(2)
        return os.environ.get(key, default if default is not None else "")
    return os.path.expandvars(value)


def expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, tuple):
        return tuple(expand_env(v) for v in value)
    if isinstance(value, str):
        return _expand_string(value)
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = expand_env(cfg)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def get_nested(cfg: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    return get_nested(cfg, dotted.split("."), default)


def update_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> Dict[str, Any]:
    cur = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--no-train", action="store_true")
    return parser.parse_args()
