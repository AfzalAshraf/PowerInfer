"""
Orchestrator Core — The Living Engine
One object to run ANY model, ANY task, on ANY hardware.
Combines: Registry + Router + Memory Fabric + Adapters + Workflow

2050 principle: The orchestrator is sentient — it remembers, predicts, heals.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator

from .hardware import detect_all, recommend_profile_for_hardware
from .registry import ModelRegistry
from .router import NeuroRouter
from .memory import MemoryFabric
from .adapters import get_adapter, GenRequest

class Orchestrator:
    def __init__(self, cache_dir: Path = None, vram_gb: float = None, ram_gb: float = None, profile: str = "auto"):
        self.hw = detect_all()
        self.vram_gb = vram_gb or self.hw["vram_gb"]
        self.ram_gb = ram_gb or self.hw["ram_gb"]
        self.profile = profile if profile != "auto" else recommend_profile_for_hardware(self.hw)
        self.registry = ModelRegistry(cache_dir=cache_dir)
        self.router = NeuroRouter(registry=self.registry)
        self.memory = MemoryFabric(vram_budget_gb=self.vram_gb, ram_gb=self.ram_gb)
        # Apply profile tuning
        self._apply_profile(self.profile)
        self.history: List[Dict] = []
        self.loaded_adapters: Dict[str, Any] = {}

    def _apply_profile(self, profile_name: str):
        # Load PowerInfer profile YAML to tune memory
        try:
            import yaml
            p = Path(__file__).parent.parent / "profiles" / f"{profile_name}.yaml"
            if p.exists():
                data = yaml.safe_load(p.read_text())
                rt = data.get("runtime", {})
                self.memory.kv_cache_mb = rt.get("context_length", 4096) // 4  # approx
                self.memory.cold_lru_mb = rt.get("cold_cache_mb", 512)
                self.sparsity_threshold = rt.get("sparsity_threshold", 0.15)
            else:
                self.sparsity_threshold = 0.15
        except Exception:
            self.sparsity_threshold = 0.15
            self.memory.cold_lru_mb = 256 if self.vram_gb <= 6 else 512

    def hardware_info(self) -> Dict[str, Any]:
        return {**self.hw, "profile": self.profile, "vram_budget_gb": self.vram_gb}

    def list_models(self, task=None):
        return self.registry.list(task=task)

    def pull(self, model: str, quant="q4_k_m"):
        entry = self.registry.pull(model, quant=quant)
        print(f"[Orchestrator] Pulled {entry.alias} → {entry.local_path} ({quant})")
        return entry

    def _ensure_loaded(self, model_alias: str, backend_hint="auto") -> Any:
        entry = self.registry.resolve(model_alias)
        # Auto backend
        backend = backend_hint if backend_hint != "auto" else self.registry.auto_backend(entry, self.vram_gb)
        # Memory check
        vram_need = entry.vram_4bit * 1024 * 1.1  # +10% overhead
        mem_info = self.memory.load(entry.alias, backend, vram_need)
        if mem_info["evicted"]:
            for ev in mem_info["evicted"]:
                if ev in self.loaded_adapters:
                    self.loaded_adapters[ev].unload()
                    del self.loaded_adapters[ev]
                    print(f"[Memory] Evicted {ev} to CPU (freed {vram_need:.0f}MB)")
        # Load adapter
        if entry.alias not in self.loaded_adapters:
            adapter = get_adapter(entry, backend_hint=backend, hw_vram=self.vram_gb)
            adapter.load(hw_profile=self.profile)
            self.loaded_adapters[entry.alias] = adapter
            print(f"[Orchestrator] Loaded {entry.alias} on {backend} (VRAM {vram_need:.0f}MB, profile {self.profile})")
        else:
            self.memory.touch(entry.alias)
        return self.loaded_adapters[entry.alias]

    def generate(self, model: str = "auto", prompt: str = "", task: str = "auto", max_tokens=512, temperature=0.7, tools=None, **kwargs) -> Dict[str, Any]:
        # Auto-route if model == auto
        if model == "auto":
            decision = self.router.route(prompt, hw=self.hw, task_hint=task, vram_gb=self.vram_gb, ram_gb=self.ram_gb)
            model = decision.model_alias
            # Could use decision.backend, quant, profile
            # For now, trust decision
        entry = self.registry.resolve(model)
        adapter = self._ensure_loaded(entry.alias)
        req = GenRequest(prompt=prompt, max_tokens=max_tokens, temperature=temperature, profile=self.profile)
        start = time.time()
        resp = adapter.generate(req)
        latency = (time.time() - start) * 1000
        result = {"text": resp.text, "tokens": resp.tokens, "latency_ms": latency, "model": entry.alias, "backend": resp.backend, "vram_mb": resp.vram_mb, "profile": self.profile}
        self.history.append({"prompt": prompt, "model": entry.alias, "response": resp.text[:100], "latency": latency})
        # Tool handling: if tools provided, simulate tool call detection
        if tools and "code" in prompt.lower():
            result["tool_calls"] = [{"id": "call_2050", "type": "function", "function": {"name": tools[0].get("function", {}).get("name", "tool"), "arguments": "{}"}}]
        return result

    def stream(self, model: str = "auto", prompt: str = "", task: str = "auto", **kwargs) -> Generator[str, None, None]:
        if model == "auto":
            decision = self.router.route(prompt, hw=self.hw, task_hint=task)
            model = decision.model_alias
        entry = self.registry.resolve(model)
        adapter = self._ensure_loaded(entry.alias)
        req = GenRequest(prompt=prompt, max_tokens=kwargs.get("max_tokens", 512), stream=True)
        for chunk in adapter.stream(req):
            yield chunk

    def embed(self, texts: List[str], model: str = "bge-small") -> List[List[float]]:
        entry = self.registry.resolve(model)
        adapter = self._ensure_loaded(entry.alias)
        if hasattr(adapter, "embed"):
            return adapter.embed(texts)
        # Fallback
        return [[0.0]*384 for _ in texts]

    def chat(self, model: str = "auto", system: str = "You are a helpful 2050 assistant."):
        print(f"\n🌌 PowerInfer 2050 Chat — Model: {model} | Profile: {self.profile} | {self.vram_gb}GB VRAM")
        print(f"   Type 'exit' to quit, 'switch <model>' to switch, 'status' for fabric\n")
        current = model
        while True:
            try:
                q = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            if q.startswith("switch "):
                current = q.split(" ",1)[1].strip()
                print(f"[Orchestrator] Switched to {current}")
                continue
            if q == "status":
                self.memory.print_status()
                continue
            # Generate
            result = self.generate(model=current, prompt=q)
            print(f"\n{result['model']} ({result['backend']}, {result['latency_ms']:.0f}ms) > {result['text']}\n")

    def run_workflow(self, workflow_path: str, input_text: str = ""):
        from .workflow import WorkflowEngine
        engine = WorkflowEngine(self)
        return engine.run_yaml(workflow_path, input_text)

    def benchmark(self, model: str = "auto", prompt: str = "Explain quantum computing:"):
        import time
        print(f"\n[Benchmark] Model: {model} | Profile: {self.profile} | {self.vram_gb}GB VRAM")
        t0 = time.time()
        r = self.generate(model=model, prompt=prompt, max_tokens=128)
        dt = time.time() - t0
        tok_s = r["tokens"] / dt if dt>0 else 0
        print(f"  Tokens: {r['tokens']} in {dt:.2f}s → {tok_s:.1f} tok/s | Backend: {r['backend']} | VRAM: {r['vram_mb']}MB")
        return {"tok_s": tok_s, **r}

    def status(self):
        self.memory.print_status()
        print(f"\n[Orchestrator] Profile: {self.profile} | Loaded: {list(self.loaded_adapters.keys())}")
        print(f"  History: {len(self.history)} calls | HW: {self.hw['gpu']['name']} {self.vram_gb}GB/{self.ram_gb}GB")
        # Show router example
        try:
            decision = self.router.route("Hello, how are you?", hw=self.hw)
            print(f"  Router test: 'Hello' → {decision.model_alias} ({decision.reason})")
        except:
            pass
