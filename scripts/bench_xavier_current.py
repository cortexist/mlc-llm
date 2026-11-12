"""Current Xavier matrix: fixed token IDs, greedy forced output, sequential backends."""

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from xavier_bench_prompts import DESIGN_NOTE, FILLER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["prepare", "mlc", "llama", "mtp"], required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--model-lib", type=Path)
    p.add_argument("--draft", type=Path)
    p.add_argument("--draft-lib", type=Path)
    p.add_argument("--runtime", type=Path, default=Path("dist/libs/xavier-mtp-runtime.so"))
    p.add_argument("--attention", choices=["baseline", "paged"], default="baseline")
    p.add_argument("--drafts", default="0,2", help="mtp: draft tokens per round, 0 first")
    p.add_argument("--context", type=int, default=16384)
    p.add_argument("--spec", choices=["none", "dflash", "mtp"], default="none")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--cases", default="161,448,910")
    p.add_argument("--output", type=Path)
    p.add_argument("--port", type=int, default=18081)
    a = p.parse_args()
    if a.backend == "prepare":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        prompts = []
        for count, text in [
            (161, DESIGN_NOTE),
            (448, "Read this and reply with one word.\n" + FILLER * 13),
            (910, "Read this and reply with one word.\n" + FILLER * 27),
        ]:

            def template(content):
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )

            original = template(text)
            empty = template("")
            body_start = empty.index(tokenizer.convert_tokens_to_ids("<turn|>"))
            prefix, suffix = empty[:body_start], empty[body_start:]
            assert original[:body_start] == prefix and original[-len(suffix) :] == suffix
            body = original[body_start : -len(suffix)]
            budget = count - len(empty)
            # Adjust body tokens directly; text retokenization can change a
            # boundary by one token. Preserve the deployed chat-template tokens.
            body = (body + tokenizer.encode(" text", add_special_tokens=False) * count)[:budget]
            ids = prefix + body + suffix
            assert len(ids) == count
            prompts.append(
                dict(
                    case=str(count),
                    tokens=ids,
                    rendered=tokenizer.decode(ids, clean_up_tokenization_spaces=False),
                    tokenizer_check_text=tokenizer.decode(
                        original, clean_up_tokenization_spaces=False
                    ),
                    tokenizer_check_tokens=original,
                    user_text=tokenizer.decode(body, clean_up_tokenization_spaces=False),
                    enable_thinking=False,
                    sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                )
            )
        a.prompts.write_text(json.dumps(prompts, indent=2) + "\n")
        print([(r["case"], len(r["tokens"])) for r in prompts], flush=True)
        return
    prompts = [r for r in json.loads(a.prompts.read_text()) if r["case"] in a.cases.split(",")]
    result = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, runs=[]
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save(row):
        result["runs"].append(row)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k
                    not in [
                        "response",
                        "tokens",
                        "text",
                        "accepted_by_round",
                        "metrics_before",
                        "metrics_after",
                    ]
                }
            ),
            flush=True,
        )

    if a.backend == "mlc":
        from mlc_llm import MLCEngine
        from mlc_llm.op.xavier_prefill import install
        from mlc_llm.serve.config import EngineConfig

        receipt = install("hybrid-width", a.model / "mlc-chat-config.json")
        engine = MLCEngine(
            str(a.model),
            model_lib=str(a.model_lib),
            device="cuda",
            mode="interactive",
            engine_config=EngineConfig(
                prefix_cache_mode="disable",
                prefill_chunk_size=512,
                max_num_sequence=1,
                max_single_sequence_length=a.context,
                max_total_sequence_length=a.context,
                speculative_mode="dflash" if a.spec == "dflash" else "disable",
                spec_draft_length=3 if a.spec == "dflash" else 0,
                additional_models=[(str(a.draft), str(a.draft_lib))] if a.spec == "dflash" else [],
            ),
        )
        assert receipt["target_caches"] == 1
        try:
            for prompt in prompts:
                for repeat in range(-a.warmups, a.repeats):
                    before = engine.metrics().metrics
                    start = time.perf_counter()
                    response = engine.completions.create(
                        prompt=prompt["tokens"],
                        temperature=0,
                        max_tokens=a.tokens,
                        extra_body={"debug_config": {"ignore_eos": True}},
                    )
                    response = response.model_dump()
                    usage = response["usage"]
                    assert usage["prompt_tokens"] == len(prompt["tokens"])
                    assert usage["completion_tokens"] == a.tokens
                    if repeat >= 0:
                        save(
                            dict(
                                case=prompt["case"],
                                repeat=repeat,
                                prompt_tokens=usage["prompt_tokens"],
                                completion_tokens=a.tokens,
                                prefill_tok_s=usage["extra"]["prefill_tokens_per_s"],
                                decode_tok_s=usage["extra"]["decode_tokens_per_s"],
                                elapsed_s=time.perf_counter() - start,
                                text=response["choices"][0]["text"],
                                response=response,
                                metrics_before=before,
                                metrics_after=engine.metrics().metrics,
                            )
                        )
        finally:
            engine.terminate()
    elif a.backend == "mtp":
        from bench_xavier_mtp import Program, Trial

        if a.attention == "paged":
            from mlc_llm.op.xavier_prefill import install

            install("paged", a.model / "mlc-chat-config.json")
        trial = Trial(Program(a.model, a.model_lib), Program(a.draft, a.draft_lib), a.runtime)
        for prompt in prompts:
            for repeat in range(-a.warmups, a.repeats):
                plain = None
                for width in map(int, a.drafts.split(",")):
                    row = trial.run(prompt["tokens"], a.tokens, width, ignore_eos=True)
                    assert row["completion_tokens"] == a.tokens
                    row.update(
                        case=prompt["case"],
                        repeat=repeat,
                        draft_tokens=width,
                        prefill_tok_s=len(prompt["tokens"]) / row["ttft_s"],
                    )
                    if width == 0:
                        plain = row["tokens"]
                    else:
                        row["matches_plain"] = row["tokens"] == plain
                    if repeat >= 0:
                        save(row)
    else:
        root = Path(__file__).resolve().parents[2]
        command = [
            str(root / "llama.cpp/build/bin/llama-server"),
            "-m",
            str(a.model),
            "-ngl",
            "99",
            "-fa",
            "on",
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "-c",
            str(a.context),
            "-b",
            "512",
            "-ub",
            "512",
            "--parallel",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(a.port),
            "-t",
            "2",
            "-tb",
            "2",
        ]
        if a.spec != "none":
            command += [
                "-md",
                str(a.draft),
                "-ngld",
                "99",
                "--spec-type",
                "draft-dflash" if a.spec == "dflash" else "draft-mtp",
                "--spec-draft-n-max",
                "2",
            ]
        result["command"] = command
        base = f"http://127.0.0.1:{a.port}"

        def request(path, body=None):
            req = urllib.request.Request(
                base + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=240) as response:
                return json.load(response)

        with a.output.with_suffix(".server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"llama-server exited {server.returncode}")
                    try:
                        request("/health")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("llama startup") from None
                        time.sleep(1)
                for prompt in prompts:
                    encoded = request(
                        "/tokenize",
                        {
                            "content": prompt["tokenizer_check_text"],
                            "add_special": False,
                            "parse_special": True,
                        },
                    )["tokens"]
                    assert encoded == prompt["tokenizer_check_tokens"], (
                        "MLC/GGUF tokenizer mismatch"
                    )
                    for repeat in range(-a.warmups, a.repeats):
                        start = time.perf_counter()
                        response = request(
                            "/completion",
                            {
                                "prompt": prompt["tokens"],
                                "n_predict": a.tokens,
                                "temperature": 0,
                                "seed": 0,
                                "cache_prompt": False,
                                "ignore_eos": True,
                            },
                        )
                        assert response["tokens_predicted"] == a.tokens, response
                        assert response["timings"]["prompt_n"] == len(prompt["tokens"]), response[
                            "timings"
                        ]
                        if repeat >= 0:
                            save(
                                dict(
                                    case=prompt["case"],
                                    repeat=repeat,
                                    prompt_tokens=len(prompt["tokens"]),
                                    completion_tokens=a.tokens,
                                    prefill_tok_s=response["timings"]["prompt_per_second"],
                                    decode_tok_s=response["timings"]["predicted_per_second"],
                                    elapsed_s=time.perf_counter() - start,
                                    text=response["content"],
                                    response=response,
                                )
                            )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
