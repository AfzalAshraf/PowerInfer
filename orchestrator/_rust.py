"""
Rust Shim — Try to use blazing fast Rust core, fallback to Python
This is how you get professional latency without losing compatibility.
"""
import sys
from pathlib import Path

HAS_RUST = False
_rust = None

try:
    # Try to import compiled Rust extension (maturin develop)
    import orchestrator_core as _rust  # type: ignore
    HAS_RUST = True
except ImportError:
    try:
        # Try alternative name
        import orchestrator_core_rs as _rust  # type: ignore
        HAS_RUST = True
    except ImportError:
        HAS_RUST = False
        _rust = None

def has_rust() -> bool:
    return HAS_RUST

def rust_version():
    if HAS_RUST and _rust:
        try:
            return _rust.version()  # type: ignore
        except:
            return "rust-unknown"
    return None

def detect_hardware_rust():
    """Call Rust hardware detect if available, else None"""
    if not HAS_RUST:
        return None
    try:
        # Rust returns dict-like
        return _rust.detect_hardware()  # type: ignore
    except Exception as e:
        print(f"[Rust Shim] hardware failed, fallback to Python: {e}", file=sys.stderr)
        return None

def router_route_rust(prompt: str, vram_gb: int, ram_gb: int, task_hint: str = "auto"):
    if not HAS_RUST:
        return None
    try:
        return _rust.route(prompt, vram_gb, ram_gb, task_hint)  # type: ignore
    except Exception as e:
        print(f"[Rust Shim] router failed: {e}", file=sys.stderr)
        return None

# For debugging
if __name__ == "__main__":
    print(f"HAS_RUST={HAS_RUST}")
    if HAS_RUST:
        print(f"version={rust_version()}")
        print(f"hw={detect_hardware_rust()}")
    else:
        print("Rust core not compiled. Run: maturin develop --features python")
        print("Falling back to Python — still works, just 10-50x slower on hot paths")
