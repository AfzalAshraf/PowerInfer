#!/usr/bin/env python3
"""
PowerInfer 2.0 — Profile Recommender
Detects your hardware (VRAM, RAM, GPU arch) and recommends the best polished profile.
Usage:
  python scripts/recommend_profile.py               # auto-detect + recommend
  python scripts/recommend_profile.py --list        # list all polished profiles
  python scripts/recommend_profile.py --profile laptop-6gb  # show one profile
  python scripts/recommend_profile.py --model 7B    # recommend for specific model size
  ./powerinfer --recommend                          # via powerinfer wrapper
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROFILES_DIR = ROOT / "profiles"

# Fallback if yaml not installed
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

def load_profile(name: str):
    """Load profile by name (without extension)"""
    # Try YAML first if available, then JSON sidecar
    for ext in [".yaml", ".yml", ".json"]:
        p = PROFILES_DIR / f"{name}{ext}"
        if p.exists():
            text = p.read_text()
            if p.suffix in (".yaml", ".yml"):
                if HAS_YAML:
                    try:
                        return yaml.safe_load(text)
                    except Exception:
                        # Fall back to JSON sidecar if YAML parse fails
                        json_sidecar = PROFILES_DIR / f"{name}.json"
                        if json_sidecar.exists():
                            return json.loads(json_sidecar.read_text())
                        return {"name": name, "display_name": name, "raw": text}
                else:
                    # No yaml lib: try JSON sidecar
                    json_sidecar = PROFILES_DIR / f"{name}.json"
                    if json_sidecar.exists():
                        return json.loads(json_sidecar.read_text())
                    # Last resort: simple parser for key fields
                    try:
                        return json.loads(text)
                    except:
                        return {"name": name, "display_name": name, "raw": text}
            else:
                return json.loads(text)
    return None

def list_profiles():
    profiles = []
    for f in PROFILES_DIR.glob("*.yaml"):
        data = load_profile(f.stem)
        if data:
            profiles.append(data)
    for f in PROFILES_DIR.glob("*.json"):
        if f.stem not in [p["name"] for p in profiles]:
            data = load_profile(f.stem)
            if data:
                profiles.append(data)
    # Sort by vram
    profiles.sort(key=lambda x: x.get("hardware", {}).get("vram_gb", 0))
    return profiles

def detect_vram_gb():
    """Try nvidia-smi, fallback to env, then 6"""
    # Try nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # May have multiple GPUs, take first
            line = result.stdout.strip().split("\n")[0]
            mb = int(line.strip())
            gb = round(mb / 1024)
            return gb
    except Exception:
        pass
    # Try env var
    if "POWERINFER_VRAM" in os.environ:
        try:
            return int(os.environ["POWERINFER_VRAM"])
        except:
            pass
    # Try /proc/driver/nvidia? fallback
    return None

def detect_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(re.search(r"\d+", line).group(0))
                    return round(kb / 1024 / 1024)
    except Exception:
        pass
    if "POWERINFER_RAM" in os.environ:
        try:
            return int(os.environ["POWERINFER_RAM"])
        except:
            pass
    return None

def detect_gpu_name():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0].strip()
    except:
        pass
    return "Unknown GPU"

def recommend(vram_gb: int = None, ram_gb: int = None, model_size: str = "7B"):
    """
    Recommend best profile for given specs.
    Model sizes: 7B, 13B, 30B, 70B
    """
    if vram_gb is None:
        vram_gb = detect_vram_gb() or 6
    if ram_gb is None:
        ram_gb = detect_ram_gb() or 16

    profiles = list_profiles()
    # Find closest profile where vram_gb <= profile vram and ram_gb <= profile ram
    # Prefer smallest profile that fits, otherwise largest that is smaller (with warning)
    candidates = [p for p in profiles if p["hardware"]["vram_gb"] >= vram_gb and p["hardware"]["ram_gb"] >= ram_gb]
    if candidates:
        # Smallest fitting
        best = min(candidates, key=lambda x: (x["hardware"]["vram_gb"], x["hardware"]["ram_gb"]))
        fit = "perfect"
    else:
        # No perfect fit, closest smaller (will need swap)
        best = max(profiles, key=lambda x: x["hardware"]["vram_gb"])
        # Actually find largest profile that is <= your hardware, to avoid recommending 32GB to 6GB user
        smaller = [p for p in profiles if p["hardware"]["vram_gb"] <= vram_gb]
        if smaller:
            best = max(smaller, key=lambda x: x["hardware"]["vram_gb"])
            fit = "closest-smaller"
        else:
            best = min(profiles, key=lambda x: x["hardware"]["vram_gb"])
            fit = "ultra-light-fallback"

    # If model_size large and RAM small, warn and suggest smaller profile
    # e.g., 70B on 16GB RAM not recommended even with 24GB VRAM -> suggest still 24gb but warn
    warning = ""
    model_gb = {"7B": 4, "13B": 8, "30B": 18, "70B": 40}.get(model_size, 4)
    if model_size in ("30B","70B") and ram_gb < 32:
        warning = f"⚠️  {model_size} ideally needs 32-64GB RAM (you have {ram_gb}GB). Will use SSD swap → 30-50% slower. Consider 7B for your RAM."

    return best, fit, warning, vram_gb, ram_gb

def print_profiles_table(profiles):
    print("\n" + "="*92)
    print(" PowerInfer 2.0 — Polished Profiles (pick one, or run --recommend for your hardware)")
    print("="*92)
    print(f"{'Profile':<18} {'VRAM':<6} {'RAM':<6} {'Hot':<5} {'ColdCache':<10} {'Best For':<30} {'Stars'}")
    print("-"*92)
    for p in profiles:
        hw = p.get("hardware", {})
        rt = p.get("runtime", {})
        rec = p.get("recommendation", {})
        name = p.get("name","")
        vram = f"{hw.get('vram_gb','')}GB"
        ram = f"{hw.get('ram_gb','')}GB"
        hot = f"{int(p.get('profiler',{}).get('hot_ratio',0)*100)}%"
        cache = f"{rt.get('cold_cache_mb','')}MB"
        best = rec.get("best_for","")[:30]
        stars = rec.get("stars","")
        # Mark recommended for 6GB user
        marker = " ← YOURS?" if name=="laptop-6gb" else ""
        print(f"{name:<18} {vram:<6} {ram:<6} {hot:<5} {cache:<10} {best:<30} {stars}{marker}")
    print("="*92)
    print("Tip: Run `python scripts/recommend_profile.py --model 7B` to get personalized pick.\n")

def print_profile_detail(profile, fit, warning, vram_gb, ram_gb, model_size="7B"):
    print("\n" + "="*92)
    gpu_name = detect_gpu_name()
    print(f" ✅ Recommended for you: {profile.get('display_name','')}")
    print(f"    Detected: {gpu_name} {vram_gb}GB VRAM + {ram_gb}GB RAM | Model: {model_size} | Fit: {fit}")
    if warning:
        print(f"    {warning}")
    print("="*92)
    print(f"Profile: {profile['name']} ({profile['hardware']['gpu']})")
    print(f"Description: {profile.get('description','')}")
    print()
    print("Profiler (profile.py):")
    pf = profile["profiler"]
    print(f"  --hot-ratio {pf['hot_ratio']} --warm-ratio {pf['warm_ratio']} --tokens {pf['tokens']}")
    print("Converter (convert_sparse):")
    cv = profile["converter"]
    print(f"  --hot-quant {cv['hot_quant']} --cold-quant {cv['cold_quant']}")
    print("Runtime (powerinfer):")
    rt = profile["runtime"]
    print(f"  --cold-cache {rt['cold_cache_mb']} --sparsity-threshold {rt['sparsity_threshold']} --vram-budget {rt.get('vram_budget_gb','auto')} -c {rt.get('context_length',4096)} {'--fallback-dense' if rt.get('fallback_dense') else ''}")
    print()
    print("Expected performance (your hardware):")
    exp = profile.get("expected", {}).get("performance", {})
    # Show for requested model size if exists
    if model_size in exp or f"{model_size}-Q4" in exp:
        key = model_size if model_size in exp else f"{model_size}-Q4"
        # Try variations
        for k in [model_size, f"{model_size}-Q4", f"{model_size}-Q4".replace("B","B-Q4")]:
            if k in exp:
                key = k
                break
        data = exp.get(key) or exp.get(model_size) or list(exp.values())[0]
        print(f"  {model_size}: {data.get('sparse_tok_s','?')} tok/s (dense {data.get('dense_tok_s','?')} → {data.get('speedup','?')} speedup) | VRAM {data.get('vram_mb','?')} MB | {data.get('quality','')}")
    else:
        for k,v in exp.items():
            print(f"  {k}: {v['sparse_tok_s']} tok/s ({v['speedup']}) | {v['quality']}")
    print()
    print("CLI shortcuts (copy-paste):")
    for k,v in profile.get("cli", {}).items():
        print(f"  {k}: {v}")
    print()
    print("Why this profile for your specs?")
    print(f"  {profile['recommendation'].get('for_specs','')}")
    print(f"  {profile['recommendation'].get('best_for','')}")
    print(f"  {profile['recommendation'].get('stars','')}")
    if profile["name"] == "laptop-6gb":
        print()
        print("  💡 For your 6GB Lenovo 15A: Start with 7B Q4_K_M. 13B works but needs -c 2048. 30B will swap to SSD.")
    print("="*92 + "\n")

def main():
    parser = argparse.ArgumentParser(description="PowerInfer 2.0 Profile Recommender")
    parser.add_argument("--list", action="store_true", help="List all polished profiles")
    parser.add_argument("--profile", type=str, help="Show detail for a profile (e.g., laptop-6gb)")
    parser.add_argument("--model", type=str, default="7B", help="Model size for recommendation (7B, 13B, 30B, 70B)")
    parser.add_argument("--vram", type=int, help="Override detected VRAM GB (for testing)")
    parser.add_argument("--ram", type=int, help="Override detected RAM GB")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--interactive", action="store_true", help="Interactive selector")
    args = parser.parse_args()

    if args.list:
        profiles = list_profiles()
        if args.json:
            print(json.dumps(profiles, indent=2))
        else:
            print_profiles_table(profiles)
        return

    if args.profile:
        p = load_profile(args.profile)
        if not p:
            print(f"Profile '{args.profile}' not found. Available: {[x.stem for x in PROFILES_DIR.glob('*.yaml')]}")
            sys.exit(1)
        # Show detail with fake fit
        print_profile_detail(p, "manual", "", args.vram or detect_vram_gb() or 6, args.ram or detect_ram_gb() or 16, args.model)
        return

    if args.interactive:
        profiles = list_profiles()
        print_profiles_table(profiles)
        print("Select a profile number:")
        for i, p in enumerate(profiles):
            print(f"  {i+1}. {p['name']} — {p['display_name']}")
        try:
            choice = int(input("Enter number (or 0 for auto-recommend): ").strip())
            if 1 <= choice <= len(profiles):
                p = profiles[choice-1]
                print_profile_detail(p, "interactive", "", args.vram or detect_vram_gb() or 6, args.ram or detect_ram_gb() or 16, args.model)
                return
        except:
            pass
        # Fall through to auto

    # Auto recommend
    vram = args.vram if args.vram is not None else detect_vram_gb()
    ram = args.ram if args.ram is not None else detect_ram_gb()
    best, fit, warning, vram_det, ram_det = recommend(vram, ram, args.model)

    if args.json:
        out = {"detected": {"vram_gb": vram_det, "ram_gb": ram_det, "gpu": detect_gpu_name()}, "model": args.model, "fit": fit, "warning": warning, "recommended": best}
        print(json.dumps(out, indent=2))
    else:
        # Show table + highlight
        profiles = list_profiles()
        print_profiles_table(profiles)
        print_profile_detail(best, fit, warning, vram_det, ram_det, args.model)

        # Also suggest next steps
        print("Next steps for your 6GB Lenovo 15A (example):")
        print(f"  python profile.py --model Qwen/Qwen2.5-7B --dataset c4-validation --tokens {best['profiler']['tokens']} --hot-ratio {best['profiler']['hot_ratio']} --warm-ratio {best['profiler']['warm_ratio']} --output qwen7b.predictor")
        print(f"  ./convert_sparse --input qwen7b.q4_k_m.gguf --predictor qwen7b.predictor --output qwen7b-sparse.gguf --hot-quant {best['converter']['hot_quant']} --cold-quant {best['converter']['cold_quant']}")
        print(f"  ./powerinfer -m qwen7b-sparse.gguf -p \"Hello\" -n 512 -c {best['runtime']['context_length']} --predictor qwen7b.predictor --cold-cache {best['runtime']['cold_cache_mb']} --sparsity-threshold {best['runtime']['sparsity_threshold']} --vram-budget {best['runtime'].get('vram_budget_gb',5)} {'--fallback-dense' if best['runtime'].get('fallback_dense') else ''} --verbose")
        print()

if __name__ == "__main__":
    main()
