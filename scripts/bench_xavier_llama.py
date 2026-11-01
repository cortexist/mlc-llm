"""Compare llama.cpp targets and draft modes with fixed prompts and output limits."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from xavier_bench_prompts import DESIGN_NOTE, FILLER

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--port", type=int, default=18081)
p.add_argument("--model", type=Path)
p.add_argument("--draft-model", type=Path)
p.add_argument("--spec-type", choices=["none", "draft-dflash", "draft-mtp"], default="draft-dflash")
p.add_argument("--draft-tokens", type=int, default=2)
p.add_argument("--cases", default="chat,448,1768")
p.add_argument("--disable-thinking", action="store_true")
args = p.parse_args()
if args.model and args.spec_type != "none" and not args.draft_model:
    p.error("A custom target requires --draft-model, or --spec-type none")
workspace = Path(__file__).resolve().parents[2]
prompts = {
    "chat": DESIGN_NOTE,
    "448": "Read this and reply with one word.\n" + FILLER * 13,
    "1768": "Read this and reply with one word.\n" + FILLER * 53,
}
models = workspace.parent / "Models/nemotron-3.5-Bolt-E4B-4GB"
model = args.model or models / "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"
draft = args.draft_model or models / "Nemotron-3.5-Bolt-E4B-DFlash-F16.gguf"
command = [str(workspace / "llama.cpp/build/bin/llama-server"), "-m", str(model)]
if args.spec_type != "none":
    command += [
        "-md",
        str(draft),
        "--spec-type",
        args.spec_type,
        "--spec-draft-n-max",
        str(args.draft_tokens),
        "-ngld",
        "99",
    ]
command += [
    "-ngl",
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
        for case in args.cases.split(","):
            prompt = prompts[case]
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
                            **(
                                {"chat_template_kwargs": {"enable_thinking": False}}
                                if args.disable_thinking
                                else {}
                            ),
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
