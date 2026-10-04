"""GPU 서버(RunPod 등)에서 임베딩만 계산해 벡터 캐시 파일로 내보낸다 — DB 불필요.

로컬 CPU로는 전량 임베딩에 하루 이상 걸려, 벡터 계산만 GPU에서 하고
적재는 로컬에서 캐시를 읽어 수행한다:

    # GPU 서버 (data-pipeline/ 에서, data/processed/*.csv 업로드 후)
    python -m embedding.export_vectors                    # → data/vectors.npz
    # 로컬 (vectors.npz 내려받은 뒤)
    EMBED_CACHE=data/vectors.npz python run_pipeline.py --skip-collect

모델·리비전·max_seq_length·정규화는 common.get_model / embed_passages 와 동일하게 쓰므로
로컬 계산 결과와 같은 벡터 공간이다. 정밀도도 fp32 그대로 둔다(fp16은 미세하게 달라짐).
"""
import argparse
import os
import time

from embedding import embed_courses, embed_jobs
from embedding.common import EMBEDDING_DIM, get_model, save_vector_cache

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "vectors.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT, help="출력 npz 경로")
    parser.add_argument("--batch-size", type=int, default=128, help="GPU 배치 크기")
    parser.add_argument("--limit", type=int, default=None, help="소스별 건수 제한 (테스트용)")
    args = parser.parse_args()

    texts = [r["required_skills_text"] for r in embed_jobs.prepare(args.limit)]
    texts += [r["content_text"] for r in embed_courses.prepare(args.limit)]
    unique = list(dict.fromkeys(texts))
    print(f"고유 텍스트 {len(unique)}건 임베딩 시작")

    model = get_model()
    print(f"디바이스: {model.device}")
    t0 = time.time()
    vecs = model.encode(unique, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    assert vecs.shape == (len(unique), EMBEDDING_DIM), vecs.shape
    print(f"임베딩 완료 {time.time() - t0:.0f}초")

    save_vector_cache(args.out, unique, vecs)
    print(f"저장: {os.path.abspath(args.out)} ({os.path.getsize(args.out) / 1e6:.0f}MB)")


if __name__ == "__main__":
    main()
