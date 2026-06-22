import os
import yaml
from typing import Dict, Any, Optional

_config: Optional[Dict[str, Any]] = None

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    global _config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    return _config

def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config

def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    def deep_update(d: Dict, u: Dict) -> Dict:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d
    _config = deep_update(_config, updates)
    return _config
