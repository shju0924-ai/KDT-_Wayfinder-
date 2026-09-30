"""임베딩 파이프라인 공용 모듈 — 로컬 임베딩 모델(bge-m3)·청킹·pgvector 적재 헬퍼.

주의: 테이블 DDL은 backend/app/db/models.py 와 항상 일치시킬 것
     (적재는 여기서 raw SQL로, 조회는 backend가 SQLAlchemy 모델로 수행).
"""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

# ── 임베딩 (로컬 BAAI/bge-m3, 1024차원) ──────────────────────
# 모델·차원은 backend/app/services/embedding.py 와 반드시 일치해야 함
# (bge-m3는 query/passage 프리픽스 구분이 없어 backend 검색어도 같은 방식으로 임베딩)
# API 키 불필요. 첫 실행 시 모델(약 2GB)을 Hugging Face에서 내려받아 캐시한다.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # backend/app/services/embedding.py 와 일치해야 함
# 입력 토큰 상한. bge-m3는 최대 8,192토큰까지 받지만 CPU에서는 길이에 비례해 느려지므로
# 512로 제한하고, 그보다 긴 텍스트는 아래 청킹으로 나눠 잘림을 막는다.
# backend/app/services/embedding.py 의 EMBED_MAX_SEQ_LENGTH 와 일치시킬 것.
EMBED_MAX_SEQ_LENGTH = 512
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

# ── 청킹 정책 ───────────────────────────────────────────────
# 채용공고·훈련과정 텍스트(수백 자 수준)는 대부분 청킹이 불필요. 예외적으로 긴 텍스트
# (모집요강 전문 등)만 문자 기준으로 분할한다. 한국어는 bge-m3 토크나이저 기준 대략
# 1~1.5자당 1토큰이라 500자면 512토큰 안에 들어간다.
# 청크는 source_id 에 "#c1", "#c2" 접미어를 붙여 별도 행으로 저장 → 검색 시 원문 단위로 묶어 해석.
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """max_chars 초과 텍스트를 overlap을 두고 분할. 짧으면 그대로 1개 반환."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += max_chars - overlap
    return chunks


_model = None


def get_model():
    """모델은 무거워 프로세스당 한 번만 로드한다."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
        _model.max_seq_length = EMBED_MAX_SEQ_LENGTH
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """텍스트 목록 → passage 임베딩 (정규화, 배치는 모델이 내부 분할).

    CPU 임베딩은 느리므로 같은 텍스트는 한 번만 계산해 결과를 재사용한다
    (채용공고는 제목·조건이 같은 공고가 많아 고유 텍스트가 원문 행보다 훨씬 적다).
    """
    unique = list(dict.fromkeys(texts))
    vecs = get_model().encode(
        unique, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=len(unique) > BATCH_SIZE
    )
    # psycopg 벡터 삽입 전 파이썬 float로 통일
    by_text = {t: [float(x) for x in v] for t, v in zip(unique, vecs)}
    return [by_text[t] for t in texts]


# ── DB (PostgreSQL + pgvector) ─────────────────────────────
DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS job_postings (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(80) UNIQUE NOT NULL,
    job_title VARCHAR(300) NOT NULL,
    company VARCHAR(200),
    region VARCHAR(20),
    source_url VARCHAR(1000),
    required_skills_text TEXT NOT NULL,
    embedding vector({EMBEDDING_DIM}),
    collected_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_job_postings_job_title ON job_postings (job_title);
CREATE INDEX IF NOT EXISTS idx_job_postings_region ON job_postings (region);
ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS source_url VARCHAR(1000);
-- 코사인 거리(<=>) 근사 검색 인덱스. 데이터 규모가 커지며(수만 건) 브루트포스
-- 풀스캔이 초 단위로 느려져 추가. HNSW는 ivfflat과 달리 빈 테이블에도 바로
-- 생성 가능하고 재작성(REINDEX) 없이 증분 삽입에 대응한다.
CREATE INDEX IF NOT EXISTS idx_job_postings_embedding_cos
    ON job_postings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS training_courses (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(80) UNIQUE NOT NULL,
    course_name VARCHAR(300) NOT NULL,
    institution VARCHAR(200),
    ncs_cd VARCHAR(20),
    ncs_nm VARCHAR(100),
    content_text TEXT NOT NULL,
    start_date VARCHAR(10),
    end_date VARCHAR(10),
    tuition INTEGER,
    address VARCHAR(200),
    embedding vector({EMBEDDING_DIM}),
    collected_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_training_courses_name ON training_courses (course_name);
CREATE INDEX IF NOT EXISTS idx_training_courses_ncs ON training_courses (ncs_cd);
CREATE INDEX IF NOT EXISTS idx_training_courses_embedding_cos
    ON training_courses USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ncs_units (
    id SERIAL PRIMARY KEY,
    ncs_code VARCHAR(30) UNIQUE NOT NULL,
    unit_name VARCHAR(500) NOT NULL,
    unit_definition TEXT,
    unit_level VARCHAR(10),
    large_category VARCHAR(200),
    middle_category VARCHAR(200),
    small_category VARCHAR(200),
    sub_category VARCHAR(200),
    source_url VARCHAR(1000),
    collected_at TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ncs_units_name ON ncs_units (unit_name);
CREATE INDEX IF NOT EXISTS idx_ncs_units_sub_category ON ncs_units (sub_category);

CREATE TABLE IF NOT EXISTS automation_occupation_scores (
    id SERIAL PRIMARY KEY,
    source VARCHAR(120) NOT NULL,
    occupation_code VARCHAR(30) NOT NULL,
    occupation_title VARCHAR(500) NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    metric VARCHAR(100) NOT NULL,
    source_version VARCHAR(100),
    source_url VARCHAR(1000) NOT NULL,
    collected_at TIMESTAMP DEFAULT now(),
    CONSTRAINT uq_automation_score_source_code UNIQUE (source, occupation_code)
);
CREATE INDEX IF NOT EXISTS idx_automation_scores_title
    ON automation_occupation_scores (occupation_title);
"""


def get_db_url() -> str:
    """SQLAlchemy 형식(postgresql+psycopg://)도 psycopg 형식으로 변환해 반환."""
    url = os.getenv("DATABASE_URL", "postgresql://wayfinder:wayfinder@localhost:5432/wayfinder")
    return url.replace("postgresql+psycopg://", "postgresql://")


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(get_db_url())
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def ensure_tables(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def truncate_table(conn: psycopg.Connection, table: str) -> None:
    """재적재 전 테이블을 비운다 — 만료돼 이번 수집에서 빠진 행이 upsert로는
    남기 때문에, 열린 공고/과정만 유지하려면 먼저 비운 뒤 새로 넣어야 한다."""
    allowed = {"job_postings", "training_courses"}
    if table not in allowed:
        raise ValueError(f"허용되지 않은 테이블: {table}")
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")
    conn.commit()


def upsert_job_postings(conn: psycopg.Connection, rows: list[dict]) -> int:
    """rows: {source_id, job_title, company, region, source_url, required_skills_text, embedding}"""
    sql = """
    INSERT INTO job_postings (
        source_id, job_title, company, region, source_url, required_skills_text, embedding
    )
    VALUES (
        %(source_id)s, %(job_title)s, %(company)s, %(region)s,
        %(source_url)s, %(required_skills_text)s, %(embedding)s
    )
    ON CONFLICT (source_id) DO UPDATE SET
        job_title = EXCLUDED.job_title,
        company = EXCLUDED.company,
        region = EXCLUDED.region,
        source_url = EXCLUDED.source_url,
        required_skills_text = EXCLUDED.required_skills_text,
        embedding = EXCLUDED.embedding,
        collected_at = now()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_training_courses(conn: psycopg.Connection, rows: list[dict]) -> int:
    """rows: {source_id, course_name, institution, ncs_cd, ncs_nm, content_text,
              start_date, end_date, tuition, address, embedding}"""
    sql = """
    INSERT INTO training_courses (source_id, course_name, institution, ncs_cd, ncs_nm,
                                  content_text, start_date, end_date, tuition, address, embedding)
    VALUES (%(source_id)s, %(course_name)s, %(institution)s, %(ncs_cd)s, %(ncs_nm)s,
            %(content_text)s, %(start_date)s, %(end_date)s, %(tuition)s, %(address)s, %(embedding)s)
    ON CONFLICT (source_id) DO UPDATE SET
        course_name = EXCLUDED.course_name,
        institution = EXCLUDED.institution,
        ncs_cd = EXCLUDED.ncs_cd,
        ncs_nm = EXCLUDED.ncs_nm,
        content_text = EXCLUDED.content_text,
        start_date = EXCLUDED.start_date,
        end_date = EXCLUDED.end_date,
        tuition = EXCLUDED.tuition,
        address = EXCLUDED.address,
        embedding = EXCLUDED.embedding,
        collected_at = now()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_ncs_units(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
    INSERT INTO ncs_units (
        ncs_code, unit_name, unit_definition, unit_level, large_category,
        middle_category, small_category, sub_category, source_url
    )
    VALUES (
        %(ncs_code)s, %(unit_name)s, %(unit_definition)s, %(unit_level)s,
        %(large_category)s, %(middle_category)s, %(small_category)s,
        %(sub_category)s, %(source_url)s
    )
    ON CONFLICT (ncs_code) DO UPDATE SET
        unit_name = EXCLUDED.unit_name,
        unit_definition = EXCLUDED.unit_definition,
        unit_level = EXCLUDED.unit_level,
        large_category = EXCLUDED.large_category,
        middle_category = EXCLUDED.middle_category,
        small_category = EXCLUDED.small_category,
        sub_category = EXCLUDED.sub_category,
        source_url = EXCLUDED.source_url,
        collected_at = now()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_automation_scores(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
    INSERT INTO automation_occupation_scores (
        source, occupation_code, occupation_title, score, metric,
        source_version, source_url
    )
    VALUES (
        %(source)s, %(occupation_code)s, %(occupation_title)s, %(score)s,
        %(metric)s, %(source_version)s, %(source_url)s
    )
    ON CONFLICT (source, occupation_code) DO UPDATE SET
        occupation_title = EXCLUDED.occupation_title,
        score = EXCLUDED.score,
        metric = EXCLUDED.metric,
        source_version = EXCLUDED.source_version,
        source_url = EXCLUDED.source_url,
        collected_at = now()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)
