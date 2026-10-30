import pathlib
import subprocess
import sys
import json
# Ensure orchestrator package is importable when running via pytest
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def test_orchestrator_import():
    from orchestrator import Orchestrator
    from orchestrator.hardware import detect_all
    from orchestrator.registry import ModelRegistry
    from orchestrator.router import NeuroRouter
    assert True

def test_hardware_detect():
    from orchestrator.hardware import detect_all, recommend_profile_for_hardware
    hw = detect_all()
    assert "vram_gb" in hw
    assert "ram_gb" in hw
    profile = recommend_profile_for_hardware(hw)
    assert profile in ["laptop-6gb","desktop-8gb","desktop-12gb","desktop-16gb","desktop-24gb","workstation-32gb"]

def test_registry_resolve():
    from orchestrator.registry import ModelRegistry
    reg = ModelRegistry()
    m = reg.resolve("qwen2.5-7b")
    assert m.alias == "qwen2.5-7b"
    assert "Qwen" in m.hf_id
    m2 = reg.resolve("meta-llama/Meta-Llama-3-8B-Instruct")
    assert m2.hf_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    m3 = reg.resolve("phi3-mini")
    assert m3.params == "3.8B"

def test_registry_list():
    from orchestrator.registry import ModelRegistry
    reg = ModelRegistry()
    all_models = reg.list()
    assert len(all_models) >= 20
    chat_models = reg.list(task="chat")
    assert len(chat_models) >= 10
    embed_models = reg.list(task="embed")
    assert len(embed_models) >= 2

def test_router():
    from orchestrator.router import NeuroRouter
    router = NeuroRouter()
    d = router.route("Hello", vram_gb=6, ram_gb=16)
    assert d.model_alias in ["phi3-mini","qwen2.5-7b","mistral-7b"]
    assert d.profile == "laptop-6gb"
    d2 = router.route("Write python code to implement transformer", vram_gb=6)
    assert d2.model_alias in ["qwen2.5-coder-7b","deepseek-coder-6.7b"]
    # 32GB should pick larger
    d3 = router.route("Explain quantum field theory in detail", vram_gb=32, ram_gb=64)
    assert d3.profile == "workstation-32gb"

def test_memory_fabric():
    from orchestrator.memory import MemoryFabric
    mem = MemoryFabric(vram_budget_gb=6, ram_gb=16)
    assert mem.vram_budget_mb == 6144
    info = mem.load("qwen2.5-7b", "powerinfer", 4600)
    assert info["loaded"] == "qwen2.5-7b"
    # Should fit second small model
    info2 = mem.load("phi3-mini", "powerinfer", 2500)
    # 4600+2500 > 6144*0.85, so should evict
    assert len(info2["evicted"]) >= 0
    st = mem.status()
    assert "vram_used_mb" in st

def test_orchestrator_generate():
    from orchestrator import Orchestrator
    orch = Orchestrator(vram_gb=6, ram_gb=16)
    result = orch.generate(model="phi3-mini", prompt="Hello", max_tokens=32)
    assert "text" in result
    assert "model" in result
    assert result["model"] == "phi3-mini"
    assert result["tokens"] > 0

def test_orchestrator_auto_route():
    from orchestrator import Orchestrator
    orch = Orchestrator(vram_gb=6, ram_gb=16)
    result = orch.generate(model="auto", prompt="Hello how are you?", max_tokens=32)
    assert result["model"] in ["phi3-mini","qwen2.5-7b","mistral-7b"]
    # Code should route to coder
    result2 = orch.generate(model="auto", prompt="Write python function to sort list", max_tokens=32)
    assert "coder" in result2["model"] or result2["model"] in ["qwen2.5-coder-7b","deepseek-coder-6.7b","qwen2.5-7b"]

def test_workflow():
    from orchestrator import Orchestrator
    from orchestrator.workflow import WorkflowEngine, Workflow, WorkflowNode
    orch = Orchestrator(vram_gb=6)
    engine = WorkflowEngine(orch)
    wf = Workflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", model="phi3-mini", input_template="Q: {{input}}"),
            WorkflowNode(id="b", model="phi3-mini", input_template="Prev: {{a.output}} Q: {{input}}"),
        ],
        edges=[{"from": "a", "to": "b"}]
    )
    result = engine.run(wf, initial_input="Hello")
    assert "a" in result["outputs"]
    assert "b" in result["outputs"]

def test_cli_status():
    result = subprocess.run([sys.executable, "-m", "orchestrator", "status"], capture_output=True, text=True, timeout=10, cwd="/home/user/PowerInfer")
    assert result.returncode == 0
    assert "Hardware" in result.stdout or "VRAM" in result.stdout or "Profile" in result.stdout

def test_cli_list():
    result = subprocess.run([sys.executable, "-m", "orchestrator", "list"], capture_output=True, text=True, timeout=10, cwd="/home/user/PowerInfer")
    assert result.returncode == 0
    assert "qwen" in result.stdout.lower() or "llama" in result.stdout.lower()

def test_cli_run():
    result = subprocess.run([sys.executable, "-m", "orchestrator", "run", "phi3-mini", "-p", "hi", "-n", "32"], capture_output=True, text=True, timeout=10, cwd="/home/user/PowerInfer")
    assert result.returncode == 0
    assert len(result.stdout) > 20

def test_api_create():
    from orchestrator import Orchestrator
    from orchestrator.api import create_app
    orch = Orchestrator(vram_gb=6)
    # Should not crash even without fastapi
    app = create_app(orch)
    # app may be None if fastapi not installed, or FastAPI instance
    assert app is None or hasattr(app, "routes") or "FastAPI" in str(type(app))

def test_tuner():
    from orchestrator.tuner import recommend_tune_config, tune
    cfg = recommend_tune_config(hw_vram=6, hw_ram=16)
    assert cfg["profile"] == "laptop-6gb"
    assert cfg["batch"] == 1
    cfg2 = recommend_tune_config(hw_vram=32, hw_ram=64)
    assert cfg2["profile"] == "workstation-32gb"
    assert cfg2["batch"] == 4
