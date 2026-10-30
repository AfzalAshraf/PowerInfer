#!/usr/bin/env python3
"""
PowerInfer 2050 — Top-level CLI entry
Not Ollama. Not LMStudio. Another Level.
Run ANY model on ANY hardware — 6GB to 32GB, one command.

Usage:
  python orchestrator.py status
  python orchestrator.py run qwen2.5-7b -p "hello"
  ./orchestrator run auto --prompt "code a game"
  ./orchestrator api --port 8000

Also: ./powerinfer orchestrator ...
"""
import sys
from pathlib import Path

# Ensure orchestrator package is importable
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.cli import main

if __name__ == "__main__":
    main()
