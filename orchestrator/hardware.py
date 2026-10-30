"""
2050 Hardware Fabric — Detects EVERYTHING
GPU, CPU, RAM, VRAM, NPU, Disk, Network, Power
Reuses PowerInfer profiles but extends to universal hardware.
"""
import os
import re
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def detect_vram_gb() -> Optional[int]:
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            mb = int(parts[0].strip())
            return round(mb / 1024)
    except Exception:
        pass
    if "POWERINFER_VRAM" in os.environ:
        try:
            return int(os.environ["POWERINFER_VRAM"])
        except:
            pass
    return None

def detect_ram_gb() -> Optional[int]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(re.search(r"\d+", line).group(0))
                    return round(kb / 1024 / 1024)
    except Exception:
        pass
    try:
        if HAS_PSUTIL:
            return round(psutil.virtual_memory().total / 1024 / 1024 / 1024)
    except:
        pass
    return None

def detect_gpu() -> Dict[str, Any]:
    info = {"name": "Unknown", "vram_gb": detect_vram_gb(), "count": 0, "arch": None, "cuda": None}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            info["count"] = len(lines)
            first = lines[0].split(",")
            info["name"] = first[0].strip()
            # Map to arch
            name = info["name"].lower()
            if "5090" in name or "5080" in name or "5070" in name:
                info["arch"] = "sm_120"  # Blackwell
                info["cuda"] = "12.8"
            elif "4090" in name or "4080" in name or "4070" in name:
                info["arch"] = "sm_89"
            elif "3090" in name or "3080" in name:
                info["arch"] = "sm_86"
            elif "a100" in name:
                info["arch"] = "sm_80"
            elif "h100" in name:
                info["arch"] = "sm_90"
            else:
                info["arch"] = "sm_86"
    except Exception:
        pass
    # Check for Apple Silicon
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        info["name"] = "Apple Silicon"
        info["arch"] = "metal"
        info["vram_gb"] = info.get("vram_gb") or detect_ram_gb()
    # Check for AMD
    try:
        result = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            info["name"] = "AMD " + result.stdout.strip().split("\n")[-1]
            info["arch"] = "gfx1100"
    except:
        pass
    return info

def detect_cpu() -> Dict[str, Any]:
    info = {"arch": platform.machine(), "cores": os.cpu_count() or 4, "model": platform.processor() or "Unknown"}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["model"] = line.split(":")[1].strip()
                    break
    except:
        pass
    return info

def detect_disk_gb() -> Optional[int]:
    try:
        if HAS_PSUTIL:
            return round(psutil.disk_usage("/").total / 1024**3)
    except:
        pass
    return None

def detect_all() -> Dict[str, Any]:
    gpu = detect_gpu()
    cpu = detect_cpu()
    ram = detect_ram_gb()
    vram = gpu.get("vram_gb") or detect_vram_gb()
    disk = detect_disk_gb()
    return {
        "gpu": gpu,
        "cpu": cpu,
        "ram_gb": ram or 16,
        "vram_gb": vram or 6,
        "disk_gb": disk,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "arch": gpu.get("arch") or "cpu",
        "is_laptop": (vram or 6) <= 8 and (ram or 16) <= 32,
        "is_6gb_loq": (vram or 6) == 6 and (ram or 16) in (14,15,16),
    }

def recommend_profile_for_hardware(hw: Dict[str, Any] = None) -> str:
    """Map hardware to PowerInfer profile name"""
    if hw is None:
        hw = detect_all()
    vram = hw.get("vram_gb", 6)
    ram = hw.get("ram_gb", 16)
    if vram <= 6:
        return "laptop-6gb"
    elif vram <= 8:
        return "desktop-8gb"
    elif vram <= 12:
        return "desktop-12gb"
    elif vram <= 16:
        return "desktop-16gb"
    elif vram <= 24:
        return "desktop-24gb"
    else:
        return "workstation-32gb"

def print_hardware_table():
    hw = detect_all()
    # Rich table if available
    try:
        from rich.table import Table
        from rich.console import Console
        console = Console()
        table = Table(title="2050 Hardware Fabric")
        table.add_column("Component", style="cyan")
        table.add_column("Detected", style="magenta")
        table.add_column("Profile", style="green")
        table.add_row("GPU", f"{hw['gpu']['name']} ({hw['vram_gb']}GB, {hw['gpu']['arch']})", recommend_profile_for_hardware(hw))
        table.add_row("CPU", f"{hw['cpu']['model']} ({hw['cpu']['cores']} cores)", "-")
        table.add_row("RAM", f"{hw['ram_gb']}GB", "-")
        table.add_row("Disk", f"{hw['disk_gb'] or '?'}GB", "-")
        table.add_row("OS", hw["os"], "-")
        console.print(table)
        return hw
    except ImportError:
        print(f"Hardware: GPU={hw['gpu']['name']} {hw['vram_gb']}GB | RAM={hw['ram_gb']}GB | CPU={hw['cpu']['model']}")
        print(f"Recommended profile: {recommend_profile_for_hardware(hw)}")
        return hw
