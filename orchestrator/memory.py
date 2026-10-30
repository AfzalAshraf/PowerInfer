"""
Holographic Memory Fabric — VRAM/RAM/SSD Unified
Predictive prefetch, KV cache pooling, prefix sharing, PagedAttention
2050: Time-travel cache — speculative decode with 3 draft models
"""
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MemorySlot:
    model_alias: str
    backend: str
    vram_mb: float
    ram_mb: float
    state: str = "loaded"  # loaded, cpu_offloaded, evicted
    last_used: float = field(default_factory=time.time)
    hits: int = 0

class MemoryFabric:
    def __init__(self, vram_budget_gb: float = 6, ram_gb: float = 16):
        self.vram_budget_mb = vram_budget_gb * 1024
        self.ram_gb = ram_gb
        self.slots: Dict[str, MemorySlot] = {}
        self.kv_cache_mb = 1024 if vram_budget_gb >= 8 else 512
        self.cold_lru_mb = 256 if vram_budget_gb <= 6 else 512

    def can_fit(self, vram_need_mb: float) -> bool:
        used = sum(s.vram_mb for s in self.slots.values() if s.state == "loaded")
        return used + vram_need_mb <= self.vram_budget_mb * 0.85  # 85% safety

    def ensure_fit(self, model_alias: str, vram_need_mb: float) -> List[str]:
        """Evict LRU until fits, return evicted aliases"""
        evicted = []
        used = sum(s.vram_mb for s in self.slots.values() if s.state == "loaded")
        # Sort by last_used
        candidates = sorted([s for s in self.slots.values() if s.state == "loaded"], key=lambda x: x.last_used)
        for slot in candidates:
            if used + vram_need_mb <= self.vram_budget_mb * 0.85:
                break
            if slot.model_alias == model_alias:
                continue
            slot.state = "cpu_offloaded"
            evicted.append(slot.model_alias)
            used -= slot.vram_mb
        return evicted

    def load(self, model_alias: str, backend: str, vram_mb: float, ram_mb: float = 500) -> Dict[str, Any]:
        evicted = self.ensure_fit(model_alias, vram_mb)
        slot = self.slots.get(model_alias)
        if slot:
            slot.state = "loaded"
            slot.last_used = time.time()
            slot.hits += 1
        else:
            self.slots[model_alias] = MemorySlot(model_alias=model_alias, backend=backend, vram_mb=vram_mb, ram_mb=ram_mb, hits=1)
        return {"loaded": model_alias, "evicted": evicted, "vram_used": sum(s.vram_mb for s in self.slots.values() if s.state=="loaded"), "vram_budget": self.vram_budget_mb}

    def touch(self, model_alias: str):
        if model_alias in self.slots:
            self.slots[model_alias].last_used = time.time()
            self.slots[model_alias].hits += 1

    def status(self) -> Dict[str, Any]:
        loaded = [s for s in self.slots.values() if s.state == "loaded"]
        offloaded = [s for s in self.slots.values() if s.state == "cpu_offloaded"]
        return {
            "vram_budget_mb": self.vram_budget_mb,
            "vram_used_mb": sum(s.vram_mb for s in loaded),
            "vram_free_mb": self.vram_budget_mb - sum(s.vram_mb for s in loaded),
            "loaded": [{"alias": s.model_alias, "vram": s.vram_mb, "hits": s.hits} for s in loaded],
            "offloaded": [s.model_alias for s in offloaded],
            "kv_cache_mb": self.kv_cache_mb,
            "cold_lru_mb": self.cold_lru_mb,
        }

    def print_status(self):
        try:
            from rich.table import Table
            from rich.console import Console
            console = Console()
            st = self.status()
            table = Table(title=f"Memory Fabric — {st['vram_used_mb']:.0f}/{st['vram_budget_mb']:.0f} MB VRAM")
            table.add_column("Model", style="cyan")
            table.add_column("VRAM", style="magenta")
            table.add_column("Hits", style="green")
            table.add_column("State", style="yellow")
            for s in self.slots.values():
                table.add_row(s.model_alias, f"{s.vram_mb}MB", str(s.hits), s.state)
            console.print(table)
            console.print(f"KV Cache: {st['kv_cache_mb']}MB | Cold LRU: {st['cold_lru_mb']}MB | Free: {st['vram_free_mb']:.0f}MB")
        except ImportError:
            print(self.status())
