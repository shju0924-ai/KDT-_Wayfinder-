#!/usr/bin/env bash
# RunPod Pod 안에서 Ollama(LLM)와 임베딩 서버를 띄운다. Pod 재시작 후 다시 실행하면 된다.
# 모델·캐시는 /workspace(볼륨)에 두어 재다운로드를 피한다. 둘 다 127.0.0.1 에만 열고
# 로컬 PC에서는 SSH 터널(-L 11434 -L 8010)로 접근한다.
set -euo pipefail
cd "$(dirname "$0")"

export OLLAMA_MODELS=/workspace/ollama
export HF_HOME=/workspace/hf-cache
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:9b}"
# 동시 요청 처리 수 — A40(44GB)에 9b(약 7GB)는 여유가 커서 여러 사용자·트랙을 한꺼번에 처리한다
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"

command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
# 시스템 pip 는 PEP 668 로 막혀 있다 — Pod의 CUDA torch 를 그대로 쓰는 venv 를 볼륨에 둔다
VENV=/workspace/venv
[ -x "$VENV/bin/python" ] || python -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -c "import fastapi, uvicorn, sentence_transformers" 2>/dev/null \
  || "$VENV/bin/pip" install -q fastapi uvicorn sentence-transformers

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  OLLAMA_HOST=127.0.0.1:11434 setsid nohup ollama serve > /workspace/ollama.log 2>&1 < /dev/null &
  sleep 3
fi
ollama pull "$OLLAMA_MODEL"

if ! curl -sf http://127.0.0.1:8010/health >/dev/null; then
  setsid nohup "$VENV/bin/uvicorn" embed_server:app --host 127.0.0.1 --port 8010 > /workspace/embed.log 2>&1 < /dev/null &
fi
for _ in $(seq 60); do curl -sf http://127.0.0.1:8010/health && break; sleep 5; done
echo
echo "SSH 터널 정보: ${RUNPOD_PUBLIC_IP:-?} -p ${RUNPOD_TCP_PORT_22:-?} (Jupyter 터미널에서만 보일 수 있음)"
