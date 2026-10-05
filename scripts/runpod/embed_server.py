"""GPU 서버(RunPod)용 임베딩 API — 백엔드의 EMBEDDING_API_URL 이 이 서버를 가리킨다.

모델·리비전·max_seq_length·정규화는 backend/app/services/embedding.py 및
data-pipeline/embedding/common.py 와 반드시 같아야 한다(DB 벡터와 같은 공간).
응답에 모델·리비전을 실어 보내고, 백엔드는 다르면 쓰지 않고 로컬 계산으로 돌아간다.

실행은 같은 폴더의 start.sh 가 맡는다 (127.0.0.1:8010 — 로컬에서는 SSH 터널로 접근).
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBED_MAX_SEQ_LENGTH = 512

model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cuda")
model.max_seq_length = EMBED_MAX_SEQ_LENGTH

app = FastAPI(title="Wayfinder embedding server")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., max_length=512)


class EmbedResponse(BaseModel):
    model: str
    revision: str
    vectors: list[list[float]]


@app.get("/health")
def health() -> dict:
    return {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    vecs = model.encode(req.texts, batch_size=64, normalize_embeddings=True)
    return EmbedResponse(
        model=EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        vectors=[[float(x) for x in v] for v in vecs],
    )
