# shellcheck shell=bash
# Generative lineup + vLLM serve-flag lookup for the BGU sweep.
# Adapted from the sibling pddl-copilot-experiments harness, trimmed to a
# no-tools QA benchmark: HF id + reasoning parser only (no --tool-call-parser).
# Keep this in sync with cfg['eval']['generative']['models'] in configs/.

CSQA_MODELS=(Qwen3.5:0.8B Qwen3.5:4B Qwen3.5:9B qwen3.6:35b gemma4:26b-a4b)
CSQA_THINK_MODES=(off on)

# Resolve canonical tag -> HF id (+ REASONING_PARSER, optional MAX_NUM_BATCHED_TOKENS).
# The reasoning parser splits the <think> trace into message.thinking for think=on;
# gemma4 has no think tokens (REASONING_PARSER=none -> flag omitted).
vllm_lookup() {
    unset MAX_NUM_BATCHED_TOKENS
    case "$1" in
        qwen3.6:35b)      HF_MODEL="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"; REASONING_PARSER="qwen3" ;;
        Qwen3.5:0.8B)     HF_MODEL="Qwen/Qwen3.5-0.8B";                 REASONING_PARSER="qwen3" ;;
        Qwen3.5:4B)       HF_MODEL="Qwen/Qwen3.5-4B";                   REASONING_PARSER="qwen3" ;;
        Qwen3.5:9B)       HF_MODEL="Qwen/Qwen3.5-9B";                   REASONING_PARSER="qwen3" ;;
        gemma4:26b-a4b)
            HF_MODEL="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
            REASONING_PARSER="none"; MAX_NUM_BATCHED_TOKENS="4096" ;;
        *) echo "Error: model '$1' not in CSQA_MODELS (${CSQA_MODELS[*]})" >&2; return 1 ;;
    esac
}

# `--reasoning-parser X`; empty when none/unset (passing `none` crashes vLLM).
vllm_reasoning_parser_flag() {
    local p="${REASONING_PARSER:-qwen3}"
    [ -n "$p" ] && [ "$p" != "none" ] && echo "--reasoning-parser $p"
}
