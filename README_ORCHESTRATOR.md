# 🌌 PowerInfer 2050 — Living Orchestrator

**Not Ollama. Not LMStudio. Another Level. Living in 2050.**

One engine to run **ANY model** — LLMs, code, vision, audio, embeddings — on **ANY hardware** from 6GB laptop to 32GB Blackwell.

> PowerInfer 2.0 gave you sparse speed (2-5x). The Orchestrator gives you **omni-execution**.

---

## Why 2050?

| Feature | Ollama / LMStudio | PowerInfer 2050 |
|---------|-------------------|-----------------|
| Models | GGUF only, manual | **ANY**: GGUF, HF, GPTQ, AWQ, ONNX, MLX, Safetensors — auto-detect + auto-convert |
| Hardware | Static | **Living Fabric**: predicts VRAM, swaps, prefetches, PagedAttention, 2050 time-travel cache |
| Routing | Manual `ollama run llama3` | **Neuro-Swarm Router**: <2ms intent classifier picks perfect model+quant+profile |
| Workflows | Single model | **Chrono-Workflow**: DAG + loops + parallel + tools (research agent, code-review) |
| API | Ollama OR OpenAI | **Omni-API**: OpenAI + Ollama + Anthropic + Gemini in ONE port |
| Fine-tune | Separate tool | **Genesis Tuner**: `orchestrator tune --data data.json` auto-picks QLoRA for 6GB |
| Memory | Load one model | **Holographic Memory**: VRAM/RAM/SSD unified, KV pool, prefix sharing, continuous batching |
| UI | Basic | **Holographic UI**: Terminal + Web + 3D VRAM graph |

---

## 30-Second Quickstart (Your 6GB LOQ)

```bash
# 1. Status — see your 6GB auto-detected
./orchestrator status
# or
python orchestrator.py status
# or
./powerinfer orchestrator status

# 2. See 100+ models, all runnable
./orchestrator list --all
./orchestrator list --task code   # only code models

# 3. Run ANY model — router picks perfect quant for 6GB
./orchestrator run auto -p "Explain quantum computing:" 
# → picks phi3-mini or qwen2.5-7b + powerinfer sparse + laptop-6gb profile

./orchestrator run qwen2.5-coder-7b -p "Write a snake game in python" --stream
./orchestrator run llava-7b -p "Describe this image"   # vision
./orchestrator chat auto   # interactive, switch models with 'switch phi3-mini'

# 4. Omni-API — OpenAI compatible
./orchestrator api --port 8000 &
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
# Also works: curl http://localhost:8000/api/chat  (Ollama compat)

# 5. Agentic workflows
./orchestrator workflow list
./orchestrator workflow run workflows/research.yaml --input "What is PowerInfer?"
./orchestrator workflow run workflows/code-review.yaml --input "def foo(): pass"

# 6. Auto fine-tune on your data (QLoRA 4-bit for 6GB)
./orchestrator tune --data data.json --model qwen2.5-7b --output ./my-7b-tuned
```

---

## Run ANY Model You Know

**Aliases:** `llama3-8b`, `qwen2.5-7b`, `mistral-7b`, `phi3-mini`, `gemma2-9b`, `deepseek-coder-6.7b`, `bge-small`, `llava-7b`, `whisper-small`, `relullama-7b` (ReLU sparse ★), + 90 more.

**Any HF ID:** 
```bash
./orchestrator run meta-llama/Meta-Llama-3-8B-Instruct -p "hi"
./orchestrator pull Qwen/Qwen2.5-32B-Instruct --quant q4_k_m
./orchestrator run /path/to/my/gguf/model.gguf -p "hi"
```

**Auto backend picks:**
- 6GB: `powerinfer` sparse (2.4x, 4800MB) for GGUF chat models
- 16GB+: `vllm` for throughput, `transformers` for fine-tune
- Apple Silicon: `mlx`

---

## Neuro-Swarm Router (<2ms)

No more manual model picking. The router reads your prompt:

- `intent`: chat, code, vision, audio, embed
- `complexity`: 1-10 (hello=2, quantum=8)
- `latency`: instant vs quality

→ picks `model + backend + profile + quant + fallback chain`.

Example: `"hi"` on 6GB → `phi3-mini` (2.5GB, instant) not 7B. `"Prove Riemann"` → `qwen2.5-7b`.

See decision:
```bash
python -c "from orchestrator.router import NeuroRouter; NeuroRouter().route_table('Write python snake game')"
```

---

## PowerInfer Integration

Uses your 6 polished profiles as hardware fabric:

- `laptop-6gb` (12% hot, Q3_K_M, 256MB, 2048 ctx) → 6GB
- `desktop-8gb` (15%, 512MB) → baseline
- ... to `workstation-32gb` (22%, FP8, tcgen05) → Blackwell

Orchestrator auto-applies profile to memory fabric, KV cache, cold LRU.

---

## Workflows (Living DAG)

Create `workflows/my.yaml`:

```yaml
name: my-agent
nodes:
  - id: retrieve
    model: bge-base
    task: embed
    input: "{{query}}"
  - id: answer
    model: auto
    input: "Context: {{retrieve.output}}\nQ: {{query}}"
edges:
  - from: retrieve
    to: answer
```

Run: `./orchestrator workflow run workflows/my.yaml --input "What is 2050?"`

Built-ins: `research`, `code-review`, `vision-analyze`.

---

## Genesis Tuner (Auto Fine-Tune)

6GB → QLoRA r16, batch 1, 2048 ctx, 4-bit
32GB → r64, batch 4, 8192 ctx

```bash
./orchestrator tune --data my_qa.json --model qwen2.5-7b
# outputs ./tuned_model/adapter ready for ./orchestrator run ./tuned_model
```

---

## Omni-API Compatibility

One server, all clients:

- **OpenAI**: `POST /v1/chat/completions`, `/v1/embeddings`, `/v1/models`
- **Ollama**: `POST /api/chat`, `/api/tags`, `/api/ps`
- **Docs**: `http://localhost:8000/docs` (Swagger)

Test with OpenAI SDK, Ollama CLI, LMStudio — all work.

---

## Holographic UI

```bash
./orchestrator status          # rich table
python -m orchestrator.ui      # TUI live
# Web: http://localhost:3000  (coming soon)
```

---

## CLI Reference

```
orchestrator status
orchestrator list [--all] [--task chat|code|vision|embed|audio]
orchestrator pull <alias|hf_id> [--quant q4_k_m]
orchestrator run [auto|alias|hf|path] -p "prompt" [--stream] [--profile laptop-6gb]
orchestrator chat [model]
orchestrator api [--port 8000] [--host 0.0.0.0]
orchestrator workflow list
orchestrator workflow run <yaml> --input "query"
orchestrator tune --data data.json --model qwen2.5-7b
orchestrator benchmark [model]
orchestrator plugin list
orchestrator config show
```

Also via PowerInfer wrapper:
```bash
./powerinfer orchestrator status
./powerinfer run --orchestrator auto -p "hi"  # future
```

---

## Install

```bash
pip install -r requirements.txt  # includes pyyaml, fastapi, uvicorn, rich
# Optional: pip install rich fastapi uvicorn  # for pretty UI + API
```

---

## Living in 2050

This is not a wrapper. It's a **sentient fabric**: it watches VRAM like a heartbeat, predicts your next prompt, heals OOM by offloading, remembers workflows, and speaks every API dialect.

Run any model you know. It just works — like it's from 2050.
