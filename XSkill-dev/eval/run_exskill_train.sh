#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

API_KEY_1="${API_KEY_1:-EMPTY}"
API_KEY_2="${API_KEY_2:-EMPTY}"

# Endpoint pool for local vLLM replicas. Prefer one single-GPU vLLM process per
# A800 when Qwen3-VL-8B fits on one GPU. NUM_WORKERS should roughly match
# endpoint_count * API_MAX_INFLIGHT_PER_ENDPOINT.
export LOCAL_VLM_ENDPOINTS="${LOCAL_VLM_ENDPOINTS:-http://127.0.0.1:8002/v1/chat/completions,http://127.0.0.1:8003/v1/chat/completions}"
PRIMARY_VLM_ENDPOINT="${LOCAL_VLM_ENDPOINTS%%,*}"

export API_ROUTER_POLICY="${API_ROUTER_POLICY:-least_inflight}"
export API_MAX_INFLIGHT_PER_ENDPOINT="${API_MAX_INFLIGHT_PER_ENDPOINT:-4}"
export API_ENDPOINT_COOLDOWN_SECONDS="${API_ENDPOINT_COOLDOWN_SECONDS:-10}"

export REASONING_END_POINTS="${REASONING_END_POINTS:-$LOCAL_VLM_ENDPOINTS}"
export VERIFIER_END_POINTS="${VERIFIER_END_POINTS:-$LOCAL_VLM_ENDPOINTS}"
export EXPERIENCE_END_POINTS="${EXPERIENCE_END_POINTS:-$LOCAL_VLM_ENDPOINTS}"
export IMAGE_SEARCH_CAPTION_ENDPOINTS="${IMAGE_SEARCH_CAPTION_ENDPOINTS:-$LOCAL_VLM_ENDPOINTS}"

# ============================================================================
# Reasoning Model
# ============================================================================

export REASONING_MODEL_NAME="${REASONING_MODEL_NAME:-qwen3-vl-8b}"

export REASONING_API_KEY="${REASONING_API_KEY:-$API_KEY_1}"
export REASONING_END_POINT="${REASONING_END_POINT:-$PRIMARY_VLM_ENDPOINT}"

# Legacy fallback. It is not needed for parallel serving when plural
# REASONING_END_POINTS is configured.
export REASONING_API_KEY_2="${REASONING_API_KEY_2:-$API_KEY_2}"
export REASONING_END_POINT_2="${REASONING_END_POINT_2:-}"

# ============================================================================
# Verifier Model
# ============================================================================
export VERIFIER_MODEL_NAME="${VERIFIER_MODEL_NAME:-$REASONING_MODEL_NAME}"
export VERIFIER_API_KEY="${VERIFIER_API_KEY:-$API_KEY_1}"
export VERIFIER_END_POINT="${VERIFIER_END_POINT:-$PRIMARY_VLM_ENDPOINT}"

# ============================================================================
# Experience Model
# ============================================================================
export EXPERIENCE_MODEL_NAME="${EXPERIENCE_MODEL_NAME:-$REASONING_MODEL_NAME}"

export EXPERIENCE_API_KEY="${EXPERIENCE_API_KEY:-$API_KEY_1}"
export EXPERIENCE_END_POINT="${EXPERIENCE_END_POINT:-$PRIMARY_VLM_ENDPOINT}"

# Legacy fallback. It is not needed for parallel serving when plural
# EXPERIENCE_END_POINTS is configured.
export EXPERIENCE_API_KEY_2="${EXPERIENCE_API_KEY_2:-$API_KEY_1}"
export EXPERIENCE_END_POINT_2="${EXPERIENCE_END_POINT_2:-}"

# Local open-source embedding by default. Set EXPERIENCE_EMBEDDING_BACKEND=api
# and fill API key/endpoint if you prefer an OpenAI-compatible embedding API.
export EXPERIENCE_EMBEDDING_BACKEND="${EXPERIENCE_EMBEDDING_BACKEND:-local}"
export EXPERIENCE_EMBEDDING_MODEL="${EXPERIENCE_EMBEDDING_MODEL:-BAAI/bge-m3}"
export EXPERIENCE_EMBEDDING_DEVICE="${EXPERIENCE_EMBEDDING_DEVICE:-cuda}"
export EXPERIENCE_EMBEDDING_API_KEY="${EXPERIENCE_EMBEDDING_API_KEY:-}"
export EXPERIENCE_EMBEDDING_ENDPOINT="${EXPERIENCE_EMBEDDING_ENDPOINT:-}"

# ============================================================================
# Function Calling Configuration
# ============================================================================
# Web/image search provider. Both web_search and image_search use Bocha.
# Reverse image search is local VLM caption -> Bocha search; no image upload.
export SEARCH_API_PROVIDER="${SEARCH_API_PROVIDER:-bocha}"
export IMAGE_SEARCH_PROVIDER="${IMAGE_SEARCH_PROVIDER:-bocha}"
export BOCHA_API_KEY="${BOCHA_API_KEY:-}"

# Visit uses local requests + trafilatura only.
export VISIT_BACKEND="${VISIT_BACKEND:-local}"

export ENABLE_FUNCTION_CALLING="${ENABLE_FUNCTION_CALLING:-true}"

# Available tools: web_search, image_search, visit, code_interpreter, zoom
export ENABLED_TOOLS="${ENABLED_TOOLS:-web_search, image_search, visit, code_interpreter, zoom}"

TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-eval/configs/tool_configs.yaml}"

IMAGE_SEARCH_MAX_CALLS="${IMAGE_SEARCH_MAX_CALLS:-0}"
WEB_SEARCH_MAX_CALLS="${WEB_SEARCH_MAX_CALLS:-3}"

# ============================================================================
# Inference Parameters
# ============================================================================

MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-12288}"
MAX_TURNS="${MAX_TURNS:-20}"
MAX_IMAGES="${MAX_IMAGES:-100}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"

# ============================================================================
# Experience Parameters
# ============================================================================

EXPERIENCE_MAX_OPS="${EXPERIENCE_MAX_OPS:-3}"
EXPERIENCE_MAX_ITEMS="${EXPERIENCE_MAX_ITEMS:-120}"

# Experience Retrieval Parameters
EXPERIENCE_RETRIEVAL_TOP_K="${EXPERIENCE_RETRIEVAL_TOP_K:-3}"
EXPERIENCE_LIBRARY="${EXPERIENCE_LIBRARY:-memory_bank/test/experiences.json}"

# ============================================================================
# Skill Parameters
# ============================================================================

SKILL_LIBRARY="${SKILL_LIBRARY:-memory_bank/test/SKILL.md}"
SKILL_MAX_LENGTH="${SKILL_MAX_LENGTH:-1000}"

# ============================================================================
# Running Settings
# ============================================================================

# SYSTEM_PROMPT_TYPE="multi_tool_agent_search"
SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-multi_tool_agent}"
# SYSTEM_PROMPT_TYPE="multi_tool_agent_code"
# SYSTEM_PROMPT_TYPE="agent_zoom"
# SYSTEM_PROMPT_TYPE="direct_cot"

EXP_NAME="${EXP_NAME:-qwen3vl8b_mixed_train_core_seed42}"

IMAGE_DIR="${IMAGE_DIR:-benchmark}"
DATA_PATH="${DATA_PATH:-benchmark/_mixed_protocol/train_core.json}"

OUTPUT_DIR="${OUTPUT_DIR:-output/xskill_accum/${EXP_NAME}}"
LOG_OUTPUT_DIR="${LOG_OUTPUT_DIR:-logs/xskill_accum/${EXP_NAME}}"

EXPERIENCE_LIBRARY="${EXPERIENCE_LIBRARY:-memory_bank/xskill_accum/${EXP_NAME}/experiences.json}"
SKILL_LIBRARY="${SKILL_LIBRARY:-memory_bank/xskill_accum/${EXP_NAME}/SKILL.md}"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ROLLOUTS_PER_SAMPLE="${ROLLOUTS_PER_SAMPLE:-2}"
EXPERIENCE_LARGE_BATCH="${EXPERIENCE_LARGE_BATCH:-32}"


mkdir -p "$(dirname "$LOG_OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$EXPERIENCE_LIBRARY")"
mkdir -p "$(dirname "$SKILL_LIBRARY")"
# ============================================================================
# Run Inference
# ============================================================================

# Optional but slower retrieval quality knobs:
#   --experience-retrieval-decomposition
#   --experience-retrieval-rewrite

python3 -u eval/infer_api.py \
    --input-file "$DATA_PATH" \
    --image-folder "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-turns "$MAX_TURNS" \
    --max-images "$MAX_IMAGES" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-completion-tokens "$MAX_COMPLETION_TOKENS" \
    --system-prompt-key "$SYSTEM_PROMPT_TYPE" \
    --num-workers "$NUM_WORKERS" \
    --tool-config-path "$TOOL_CONFIG_PATH" \
    --max-samples "$MAX_SAMPLES" \
    --rollouts-per-sample "$ROLLOUTS_PER_SAMPLE" \
    --image-search-max-calls "$IMAGE_SEARCH_MAX_CALLS" \
    --web-search-max-calls "$WEB_SEARCH_MAX_CALLS" \
    --skill-enable \
    --skill-library "$SKILL_LIBRARY" \
    --skill-inference \
    --experience-enable \
    --experience-library "$EXPERIENCE_LIBRARY" \
    --experience-retrieval \
    --experience-retrieval-top-k "$EXPERIENCE_RETRIEVAL_TOP_K" \
    --experience-online-generate \
    --experience-library-update \
    --experience-max-ops "$EXPERIENCE_MAX_OPS" \
    --experience-large-batch "$EXPERIENCE_LARGE_BATCH" \
    --experience-refine \
    --experience-max-items "$EXPERIENCE_MAX_ITEMS" \
    --skill-refine \
    --skill-max-length "$SKILL_MAX_LENGTH" \
    --skip-completed \
    2>&1 | tee "$LOG_OUTPUT_DIR.log"
