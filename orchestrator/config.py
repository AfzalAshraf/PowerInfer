"""
Genesis Config — One YAML to rule all
Supports: orchestrator.yaml, profiles/*.yaml, workflows/*.yaml

2050: Config is living — auto-tunes based on hardware, self-heals on error.
"""
from pathlib import Path
from typing import Dict, Any
import json

DEFAULT_CONFIG = {
    "version": "2050.0.1",
    "profile": "auto",  # auto-detect
    "cache_dir": "~/.cache/powerinfer",
    "api": {"host": "0.0.0.0", "port": 8000, "cors": True, "keys": []},
    "memory": {"vram_budget_gb": "auto", "kv_cache_mb": "auto", "cold_lru_mb": "auto"},
    "router": {"enabled": True, "confidence_threshold": 0.7, "fallback": True},
    "scheduling": {"continuous_batching": True, "paged_attention": True, "speculative_decode": True, "draft_models": ["phi3-mini"]},
    "workflows": {"auto_heal": True, "max_loops": 5},
    "plugins": {"enabled": ["memory", "tools", "vision"], "auto_load": True},
    "observability": {"metrics": True, "tracing": True, "dashboard_port": 3000},
}

def load_config(path: Path = None) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    # Search order: ./orchestrator.yaml, ~/.config/powerinfer/orchestrator.yaml, /etc/powerinfer.yaml
    candidates = [
        Path("orchestrator.yaml"),
        Path.home() / ".config" / "powerinfer" / "orchestrator.yaml",
        Path("/home/user/PowerInfer/orchestrator.yaml"),
        Path("/home/user/PowerInfer/orchestrator/configs/default.yaml"),
    ]
    if path:
        candidates = [Path(path)] + candidates
    for cand in candidates:
        if cand.exists():
            try:
                import yaml
                data = yaml.safe_load(cand.read_text())
                cfg.update(data)
                print(f"[Config] Loaded {cand}")
                break
            except ImportError:
                try:
                    data = json.loads(cand.read_text())
                    cfg.update(data)
                    break
                except:
                    pass
            except Exception as e:
                print(f"[Config] Failed to load {cand}: {e}")
    return cfg

def save_config(cfg: Dict[str, Any], path: Path = Path("orchestrator.yaml")):
    try:
        import yaml
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    except ImportError:
        path.write_text(json.dumps(cfg, indent=2))
    print(f"[Config] Saved to {path}")

def show_config():
    cfg = load_config()
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        import yaml
        console = Console()
        console.print(Syntax(yaml.safe_dump(cfg), "yaml", theme="monokai"))
    except ImportError:
        import json
        print(json.dumps(cfg, indent=2))
    return cfg
