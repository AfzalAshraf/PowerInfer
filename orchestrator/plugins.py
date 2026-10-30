"""
Plugin Fabric — Extend the orchestrator like it's 2050
Plugins: memory, tools, vision, audio, web, code-interpreter, database, etc.

Each plugin is a Python file in orchestrator/plugins/ with:
  def setup(orchestrator): ...
  def tool_spec(): return {...}
"""
from pathlib import Path
from typing import Dict, List, Any
import importlib.util
import sys

class PluginManager:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.plugins: Dict[str, Any] = {}
        self.plugin_dir = Path(__file__).parent / "plugins"

    def discover(self) -> List[str]:
        if not self.plugin_dir.exists():
            return []
        return [p.stem for p in self.plugin_dir.glob("*.py") if p.stem != "__init__"]

    def load(self, name: str) -> bool:
        path = self.plugin_dir / f"{name}.py"
        if not path.exists():
            print(f"[Plugin] Not found: {name}")
            return False
        try:
            spec = importlib.util.spec_from_file_location(f"orchestrator.plugins.{name}", str(path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            if hasattr(mod, "setup"):
                mod.setup(self.orch)
            self.plugins[name] = mod
            print(f"[Plugin] Loaded {name}")
            return True
        except Exception as e:
            print(f"[Plugin] Failed to load {name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_all(self):
        for name in self.discover():
            self.load(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for name, mod in self.plugins.items():
            if hasattr(mod, "tool_spec"):
                try:
                    spec = mod.tool_spec()
                    if isinstance(spec, list):
                        tools.extend(spec)
                    else:
                        tools.append(spec)
                except:
                    pass
        return tools

# Built-in plugin stubs — create if not exists
def ensure_builtin_plugins():
    plugin_dir = Path(__file__).parent / "plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    # Memory plugin
    mem_plugin = plugin_dir / "memory.py"
    if not mem_plugin.exists():
        mem_plugin.write_text('''"""
Memory plugin — long-term memory for agent
"""
def setup(orch):
    orch.memory_plugin = {"store": []}
    print("[Plugin:memory] Long-term memory enabled")

def tool_spec():
    return {"type": "function", "function": {"name": "remember", "description": "Store a memory", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}}
''')
    # Web search plugin
    web_plugin = plugin_dir / "web.py"
    if not web_plugin.exists():
        web_plugin.write_text('''"""
Web search plugin — simulated
"""
def setup(orch):
    print("[Plugin:web] Web search (simulated) enabled")
def tool_spec():
    return {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
''')
    # Code interpreter
    code_plugin = plugin_dir / "code.py"
    if not code_plugin.exists():
        code_plugin.write_text('''"""
Code interpreter plugin
"""
def setup(orch):
    print("[Plugin:code] Code interpreter enabled")
def tool_spec():
    return {"type": "function", "function": {"name": "run_python", "description": "Execute python code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}
''')

ensure_builtin_plugins()
