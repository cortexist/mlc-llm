#!/usr/bin/env bash
# Run all backend/mode combinations sequentially on the initialized Xavier.
set -euo pipefail
source scripts/xavier/env.sh
unset LD_PRELOAD MLC_XAVIER_GEMM_PROBE
bench_log=${MLC_CURRENT_BENCH_LOG:-logs/current-matrix-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$bench_log"
trap 'sudo -n systemctl reset-failed mlc-nemotron; sudo -n systemctl start mlc-nemotron' EXIT
sudo -n systemctl stop mlc-nemotron
run() {
    local label=$1
    shift
    timeout -k 20s 600 python scripts/bench_xavier_current.py "$@" \
        --output "$bench_log/$label.json" > "$bench_log/$label.log" 2>&1
}
python scripts/bench_xavier_current.py --backend prepare \
    --tokenizer "$HOME/Models/nemotron-3.5-Bolt-E4B-BF16" --prompts "$bench_log/nemotron-prompts.json"
python scripts/bench_xavier_current.py --backend prepare \
    --tokenizer "$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic" --prompts "$bench_log/gemma4-prompts.json"
nemo=(--prompts "$bench_log/nemotron-prompts.json" --context 16384)
gemma=(--prompts "$bench_log/gemma4-prompts.json" --context 4096)
# A short first pass catches startup, tokenizer and timing-schema errors.
run pilot-mlc --backend mlc "${nemo[@]}" --model dist/target-16k \
    --model-lib dist/libs/target-dflash-vector-dequant-16k-sm72.so \
    --tokens 8 --warmups 0 --repeats 1 --cases 161
run pilot-llama --backend llama "${nemo[@]}" \
    --model "$HOME/Models/nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf" \
    --tokens 8 --warmups 0 --repeats 1 --cases 161
run pilot-llama12b-mtp --backend llama "${gemma[@]}" --spec mtp \
    --model "$HOME/Models/gemma4-12b-heretic-gguf/target-Q4_K_M.gguf" \
    --draft "$HOME/Models/gemma4-12b-heretic-gguf/assistant-f16.gguf" \
    --tokens 8 --warmups 0 --repeats 1 --cases 161
for spec in none dflash; do
    run "nemotron-mlc-$spec" --backend mlc "${nemo[@]}" --spec "$spec" \
        --model dist/target-16k --model-lib dist/libs/target-dflash-vector-dequant-16k-sm72.so \
        --draft dist/draft-16k --draft-lib dist/libs/draft-cached-tc-16k-sm72.so
    run "nemotron-llama-$spec" --backend llama "${nemo[@]}" --spec "$spec" \
        --model "$HOME/Models/nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf" \
        --draft "$HOME/Models/nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-DFlash-F16.gguf"
done
run gemma4-mlc-mtp --backend mtp "${gemma[@]}" \
    --model dist/gemma4-12b-mtp-target-4k \
    --model-lib dist/libs/gemma4-12b-mtp-target-4k-sm72.so \
    --draft "$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant-q0f16-MLC" \
    --draft-lib dist/libs/gemma4-12b-mtp-assistant-4k-sm72.so
for spec in none mtp; do
    run "gemma4-llama-$spec" --backend llama "${gemma[@]}" --spec "$spec" \
        --model "$HOME/Models/gemma4-12b-heretic-gguf/target-Q4_K_M.gguf" \
        --draft "$HOME/Models/gemma4-12b-heretic-gguf/assistant-f16.gguf"
done
