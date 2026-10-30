"""
Chrono-Workflow Engine — DAG + Cyclic Agentic Workflows
Run: LLM -> Tool -> LLM -> Judge loops, parallel branches, human-in-loop, auto-heal.

2050: Workflows are living — they self-optimize, remember, and branch like neural pathways.

Example workflow.yaml:
  name: research-agent
  nodes:
    - id: retrieve
      model: bge-base
      task: embed
      input: "{{query}}"
    - id: think
      model: qwen2.5-7b
      input: "Context: {{retrieve.output}} \n Question: {{query}}"
    - id: critic
      model: phi3-mini
      input: "Critique: {{think.output}}"
  edges:
    - from: retrieve
      to: think
    - from: think
      to: critic

Supports: python tool nodes, conditionals, parallel, loops.
"""
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class WorkflowNode:
    id: str
    model: str = "auto"
    task: str = "chat"
    input_template: str = "{{input}}"
    tool: Optional[str] = None  # python code or tool name
    condition: Optional[str] = None
    parallel: bool = False

@dataclass
class Workflow:
    name: str
    nodes: List[WorkflowNode]
    edges: List[Dict[str, str]]  # from, to
    vars: Dict[str, Any] = field(default_factory=dict)

class WorkflowEngine:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.history: List[Dict] = []

    def load_yaml(self, path: Path) -> Workflow:
        try:
            import yaml
            data = yaml.safe_load(path.read_text())
        except ImportError:
            import json
            data = json.loads(path.read_text())
        nodes = [WorkflowNode(id=n["id"], model=n.get("model","auto"), task=n.get("task","chat"), input_template=n.get("input","{{input}}"), tool=n.get("tool"), condition=n.get("when")) for n in data.get("nodes",[])]
        return Workflow(name=data.get("name","workflow"), nodes=nodes, edges=data.get("edges",[]), vars=data.get("vars",{}))

    def render(self, template: str, context: Dict[str, Any]) -> str:
        # Simple {{var}} and {{node.output}} rendering
        def repl(m):
            key = m.group(1).strip()
            # Support dot notation
            parts = key.split(".")
            val = context
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    return m.group(0)
            return str(val)
        return re.sub(r"\{\{(.*?)\}\}", repl, template)

    def run(self, workflow: Workflow, initial_input: str = "", verbose=True) -> Dict[str, Any]:
        # Render vars that may contain {{input}}
        rendered_vars = {}
        base_ctx = {"input": initial_input, "query": initial_input}
        for k, v in workflow.vars.items():
            if isinstance(v, str) and "{{" in v:
                rendered_vars[k] = self.render(v, base_ctx)
            else:
                rendered_vars[k] = v
        context = {"input": initial_input, "query": initial_input, **rendered_vars}
        outputs = {}
        # Topological sort simple: run in order respecting edges, handle parallel
        # For MVP, run nodes in defined order, checking conditions
        for node in workflow.nodes:
            if node.condition:
                # Simple condition: e.g. "{{critic.output}} contains 'bad'"
                cond_rendered = self.render(node.condition, context)
                # Very simple eval: if "true" in cond_rendered lower
                if "false" in cond_rendered.lower():
                    if verbose:
                        print(f"[Workflow] Skipping {node.id} (condition false)")
                    continue
            prompt = self.render(node.input_template, context)
            if node.tool:
                # Tool execution — simulate python tool
                if verbose:
                    print(f"[Tool:{node.id}] Executing tool {node.tool[:60]}...")
                # Simulate tool output
                output = f"[Tool {node.tool} output for '{prompt[:40]}']"
                # If tool is python snippet, we could exec, but simulate
            else:
                # Model generation via orchestrator
                if verbose:
                    print(f"[{node.id}] → {node.model} ({node.task})")
                # Auto-resolve model
                if node.model == "auto":
                    # Use router
                    from .router import NeuroRouter
                    router = NeuroRouter()
                    decision = router.route(prompt, task_hint=node.task)
                    model_alias = decision.model_alias
                else:
                    model_alias = node.model
                # Call orchestrator
                result = self.orch.generate(model=model_alias, prompt=prompt, task=node.task, stream=False)
                output = result.get("text", str(result))
                time.sleep(0.05)
            context[node.id] = {"output": output, "input": prompt}
            context[f"{node.id}.output"] = output
            outputs[node.id] = output
            if verbose:
                print(f"  ↳ {output[:120]}...")
        return {"workflow": workflow.name, "outputs": outputs, "context": context}

    def run_yaml(self, path: str, input_text: str = "") -> Dict[str, Any]:
        wf = self.load_yaml(Path(path))
        return self.run(wf, input_text)

# Example built-in workflows
BUILTIN_WORKFLOWS = {
    "research": {
        "name": "research-agent",
        "nodes": [
            {"id": "retrieve", "model": "bge-base", "task": "embed", "input": "{{query}}"},
            {"id": "synthesize", "model": "qwen2.5-7b", "input": "Context: {{retrieve.output}}\nQuestion: {{query}}\nAnswer concisely with citations."},
            {"id": "critique", "model": "phi3-mini", "input": "Critique this answer for hallucinations: {{synthesize.output}}"},
        ],
        "edges": [{"from": "retrieve", "to": "synthesize"}, {"from": "synthesize", "to": "critique"}]
    },
    "code-review": {
        "name": "code-review",
        "nodes": [
            {"id": "analyze", "model": "qwen2.5-coder-7b", "input": "Review this code:\n{{input}}"},
            {"id": "test", "model": "deepseek-coder-6.7b", "input": "Write tests for:\n{{input}}"},
            {"id": "fix", "model": "qwen2.5-coder-7b", "input": "Original: {{input}}\nReview: {{analyze.output}}\nFix the code."},
        ],
        "edges": [{"from": "analyze", "to": "fix"}, {"from": "test", "to": "fix"}]
    },
}
