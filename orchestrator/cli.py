"""
Orchestrator CLI — The 2050 Command Center
All commands: status, list, pull, run, chat, api, workflow, tune, benchmark, plugin, config
"""
import argparse
import sys
import json
from pathlib import Path

from .core import Orchestrator
from .hardware import detect_all, print_hardware_table, recommend_profile_for_hardware

def cmd_status(args):
    hw = print_hardware_table()
    orch = Orchestrator(vram_gb=args.vram, ram_gb=args.ram, profile=args.profile)
    orch.status()

def cmd_list(args):
    orch = Orchestrator()
    if args.all:
        orch.registry.to_table(task=args.task)
    else:
        orch.registry.to_table(task=args.task)
        print("\nTip: --all to see 100+ models | --task code/vision/embed to filter")
        print("     orchestrator pull <alias>  to cache")
        print("     orchestrator run <model> --prompt 'hi'")

def cmd_pull(args):
    orch = Orchestrator()
    entry = orch.pull(args.model, quant=args.quant)
    print(f"✅ Pulled {entry.alias} ({entry.hf_id}) → {entry.local_path}")

def cmd_run(args):
    orch = Orchestrator(profile=args.profile, vram_gb=args.vram)
    if args.stream:
        print(f"\n[{args.model}] streaming...\n")
        for chunk in orch.stream(model=args.model, prompt=args.prompt, max_tokens=args.max_tokens):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        result = orch.generate(model=args.model, prompt=args.prompt, max_tokens=args.max_tokens, temperature=args.temperature, task="auto")
        print(f"\n[{result['model']}] ({result['backend']}, {result['latency_ms']:.0f}ms, {result['vram_mb']:.0f}MB)\n{result['text']}\n")
        if args.verbose:
            print(f"  Profile: {result['profile']} | Tokens: {result['tokens']}")

def cmd_chat(args):
    orch = Orchestrator(profile=args.profile, vram_gb=args.vram)
    orch.chat(model=args.model)

def cmd_api(args):
    orch = Orchestrator(profile=args.profile)
    from .api import run_api
    run_api(orch, host=args.host, port=args.port)

def cmd_workflow(args):
    orch = Orchestrator(profile=args.profile)
    if args.action == "list":
        from .workflow import BUILTIN_WORKFLOWS
        print("Built-in workflows:")
        for k, v in BUILTIN_WORKFLOWS.items():
            print(f"  {k:15} {v['name']} — {len(v['nodes'])} nodes")
        print("\nCustom workflows in workflows/*.yaml")
        for p in Path("workflows").glob("*.yaml"):
            print(f"  {p.stem:15} {p}")
        if Path("/home/user/PowerInfer/workflows").exists():
            for p in Path("/home/user/PowerInfer/workflows").glob("*.yaml"):
                print(f"  {p.stem:15} {p}")
    elif args.action == "run":
        result = orch.run_workflow(args.workflow, args.input)
        print(f"\n✅ Workflow {result['workflow']} done")
        for nid, out in result["outputs"].items():
            print(f"\n--- {nid} ---\n{out[:500]}")
    else:
        print("Unknown workflow action, use list or run")

def cmd_tune(args):
    from .tuner import tune
    tune(data_path=args.data, model=args.model, profile=args.profile, output=args.output)

def cmd_benchmark(args):
    orch = Orchestrator(profile=args.profile)
    orch.benchmark(model=args.model)

def cmd_plugin(args):
    orch = Orchestrator()
    from .plugins import PluginManager
    pm = PluginManager(orch)
    if args.action == "list":
        print("Discovered plugins:", pm.discover())
        print("Loaded:", list(pm.plugins.keys()))
        pm.load_all()
        print("Tools:", pm.list_tools())
    elif args.action == "install":
        print(f"[Plugin] Install {args.name} — copy to orchestrator/plugins/{args.name}.py")
    else:
        print("plugin list | plugin install <name>")

def cmd_config(args):
    from .config import load_config, show_config
    if args.action == "show":
        show_config()
    elif args.action == "path":
        print(Path("orchestrator.yaml").resolve())
    else:
        show_config()

def build_parser():
    parser = argparse.ArgumentParser(prog="orchestrator", description="PowerInfer 2050 — Living Orchestrator (Not Ollama, Not LMStudio, Another Level)")
    parser.add_argument("--profile", type=str, default="auto", help="Hardware profile: laptop-6gb, desktop-8gb, auto")
    parser.add_argument("--vram", type=int, default=None, help="Override VRAM GB")
    parser.add_argument("--ram", type=int, default=None, help="Override RAM GB")
    sub = parser.add_subparsers(dest="command")

    # status
    p_status = sub.add_parser("status", help="Show hardware, memory, loaded models")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = sub.add_parser("list", help="List models (zoo)")
    p_list.add_argument("--all", action="store_true", help="Show all 100+")
    p_list.add_argument("--task", type=str, default=None, help="Filter: chat, code, vision, embed, audio")
    p_list.set_defaults(func=cmd_list)

    # pull
    p_pull = sub.add_parser("pull", help="Pull/cache a model")
    p_pull.add_argument("model", type=str, help="Alias or HF id (e.g., qwen2.5-7b, meta-llama/Meta-Llama-3-8B)")
    p_pull.add_argument("--quant", type=str, default="q4_k_m", help="Quant: q4_k_m, q8_0, q3_k_m")
    p_pull.set_defaults(func=cmd_pull)

    # run
    p_run = sub.add_parser("run", help="Run ANY model with ANY prompt")
    p_run.add_argument("model", type=str, nargs="?", default="auto", help="Model alias, HF id, or 'auto' (router picks)")
    p_run.add_argument("--prompt", "-p", type=str, default="Explain quantum computing:", help="Prompt")
    p_run.add_argument("--max-tokens", "-n", type=int, default=512)
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--stream", action="store_true", help="Stream tokens")
    p_run.add_argument("--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat (like ollama run)")
    p_chat.add_argument("model", type=str, nargs="?", default="auto")
    p_chat.set_defaults(func=cmd_chat)

    # api
    p_api = sub.add_parser("api", help="Start Omni-API server (OpenAI+Ollama compat)")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--host", type=str, default="0.0.0.0")
    p_api.set_defaults(func=cmd_api)

    # workflow
    p_wf = sub.add_parser("workflow", help="Chrono-Workflow engine")
    p_wf.add_argument("action", type=str, choices=["list", "run"], help="list or run")
    p_wf.add_argument("workflow", type=str, nargs="?", help="Path to workflow.yaml or builtin name")
    p_wf.add_argument("--input", type=str, default="Hello", help="Initial input")
    p_wf.set_defaults(func=cmd_workflow)

    # tune
    p_tune = sub.add_parser("tune", help="Genesis Auto-Tuner — QLoRA on any data")
    p_tune.add_argument("--data", type=str, required=True, help="Path to data.json")
    p_tune.add_argument("--model", type=str, default="qwen2.5-7b", help="Base model")
    p_tune.add_argument("--output", type=str, default="./tuned_model")
    p_tune.set_defaults(func=cmd_tune)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark model tok/s")
    p_bench.add_argument("model", type=str, nargs="?", default="auto")
    p_bench.set_defaults(func=cmd_benchmark)

    # plugin
    p_plugin = sub.add_parser("plugin", help="Plugin fabric")
    p_plugin.add_argument("action", type=str, choices=["list", "install"])
    p_plugin.add_argument("name", type=str, nargs="?")
    p_plugin.set_defaults(func=cmd_plugin)

    # config
    p_cfg = sub.add_parser("config", help="Show/edit config")
    p_cfg.add_argument("action", type=str, nargs="?", default="show", choices=["show", "path"])
    p_cfg.set_defaults(func=cmd_config)

    return parser

def main():
    parser = build_parser()
    # If no args, show help + status
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*70)
        print("2050 Quickstart:")
        print("  orchestrator status                    # see your hardware")
        print("  orchestrator list --all                # see 100+ models")
        print("  orchestrator run auto -p 'hi'          # router picks perfect model")
        print("  orchestrator run qwen2.5-7b -p 'code a snake game'")
        print("  orchestrator chat auto                 # interactive")
        print("  orchestrator api --port 8000           # OpenAI-compatible server")
        print("  orchestrator workflow list             # agentic workflows")
        print("="*70 + "\n")
        # Show status as demo
        try:
            orch = Orchestrator()
            orch.status()
        except:
            pass
        sys.exit(0)
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)

if __name__ == "__main__":
    main()
