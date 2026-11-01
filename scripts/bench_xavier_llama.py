"""Fresh llama.cpp+DFlash control using the same prompt text and output limits."""

import argparse
import ast
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--port", type=int, default=18081)
args = p.parse_args()
workspace = Path(__file__).resolve().parents[2]
constants = {}
for node in ast.parse((workspace / "mlc-jp5/mlc_bench.py").read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ("DESIGN_NOTE", "FILLER"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
prompts = {
    "chat": constants["DESIGN_NOTE"],
    "448": "Read this and reply with one word.\n" + constants["FILLER"] * 13,
    "1768": "Read this and reply with one word.\n" + constants["FILLER"] * 53,
}
models = workspace.parent / "Models/nemotron-3.5-Bolt-E4B-4GB"
command = [
    str(workspace / "llama.cpp/build/bin/llama-server"),
    "-m",
    str(models / "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"),
    "-md",
    str(models / "Nemotron-3.5-Bolt-E4B-DFlash-F16.gguf"),
    "--spec-type",
    "draft-dflash",
    "--spec-draft-n-max",
    "2",
    "-ngl",
    "99",
    "-ngld",
    "99",
    "-fa",
    "on",
    "--jinja",
    "-c",
    "8192",
    "--parallel",
    "1",
    "--host",
    "127.0.0.1",
    "--port",
    str(args.port),
]
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
data = dict(command=command, runs=[])
base = f"http://127.0.0.1:{args.port}"
with out.with_suffix(".server.log").open("w") as log:
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited {server.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("llama-server startup")
            time.sleep(1)
        for case, prompt in prompts.items():
            for repeat in [-1, 0, 1]:
                request = urllib.request.Request(
                    base + "/v1/chat/completions",
                    data=json.dumps(
                        dict(
                            messages=[dict(role="user", content=prompt)],
                            max_tokens=128,
                            temperature=0,
                            cache_prompt=False,
                            ignore_eos=True,
                        )
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                elapsed = time.perf_counter() - start
                assert result["usage"]["completion_tokens"] == 128, result
                if repeat >= 0:
                    row = dict(case=case, repeat=repeat, elapsed_s=elapsed, response=result)
                    data["runs"].append(row)
                    out.write_text(json.dumps(data, indent=2) + "\n")
                    print(
                        json.dumps(dict(case=case, repeat=repeat, timings=result.get("timings"))),
                        flush=True,
                    )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
