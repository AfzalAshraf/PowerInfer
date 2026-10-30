"""
Holographic UI — Terminal + Web
2050: Neural dashboard, token river, 3D VRAM graph
"""
import time
from pathlib import Path

def launch_tui(orchestrator):
    # Rich TUI
    try:
        from rich.live import Live
        from rich.table import Table
        from rich.console import Console
        from rich.layout import Layout
        console = Console()
        console.print("[bold cyan]🌌 PowerInfer 2050 — Holographic UI[/bold cyan]")
        console.print("Press Ctrl+C to exit")
        # Simple live status
        for i in range(5):
            hw = orchestrator.hardware_info()
            mem = orchestrator.memory.status()
            table = Table(title=f"Live — {i+1}/5")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("VRAM", f"{mem['vram_used_mb']:.0f}/{mem['vram_budget_mb']:.0f} MB")
            table.add_row("Models", ", ".join([m['alias'] for m in mem['loaded']]) or "none")
            table.add_row("Profile", hw["profile"])
            console.print(table)
            time.sleep(1)
        return
    except ImportError:
        print("Rich not installed, falling back to simple UI")
        orchestrator.status()

def launch_web(orchestrator, port=3000):
    # Simple HTTP server with dashboard
    html = """
<!doctype html>
<html><head><title>PowerInfer 2050</title>
<style>body{font-family:monospace;background:#0a0a0f;color:#0ff;padding:40px} 
.card{border:1px solid #0ff;padding:20px;margin:20px 0;border-radius:12px;background:#111}
h1{color:#f0f;text-shadow:0 0 10px #f0f}
</style></head>
<body>
<h1>🌌 PowerInfer 2050 — Living Orchestrator</h1>
<div class="card">
<h2>Hardware Fabric</h2>
<p id="hw">Loading...</p>
</div>
<div class="card">
<h2>Memory Fabric</h2>
<p id="mem">Loading...</p>
</div>
<div class="card">
<h2>Try API</h2>
<pre>curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'</pre>
</div>
<script>
fetch('/health').then(r=>r.json()).then(j=>{document.getElementById('hw').innerText=JSON.stringify(j.hardware,null,2)})
fetch('/health').then(r=>r.json()).then(j=>{document.getElementById('mem').innerText=JSON.stringify(j.memory,null,2)})
</script>
</body></html>
"""
    # Serve via http.server
    import http.server, socketserver, threading
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
            else:
                self.send_error(404)
    print(f"🌐 Holographic Web UI on http://localhost:{port}")
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Serving at port {port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        pass
