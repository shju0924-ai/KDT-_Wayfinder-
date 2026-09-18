"""기존 job_postings 행에 원본 채용공고 URL만 채운다.

임베딩을 다시 생성하지 않고 processed CSV의 식별자와 URL을 이용한다.

사용:
    python -m embedding.backfill_job_urls
"""

from embedding.common import ensure_tables, get_connection
from embedding.embed_jobs import load_gg, load_seoul


def run() -> int:
    rows = load_seoul() + load_gg()
    url_rows = [
        {"source_id": row["source_id"], "source_url": row["source_url"]}
        for row in rows
        if row.get("source_url")
    ]
    if not url_rows:
        print("백필할 채용공고 URL이 없습니다.")
        return 0

    conn = get_connection()
    try:
        ensure_tables(conn)
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE job_postings
                SET source_url = %(source_url)s
                WHERE split_part(source_id, '#', 1) = %(source_id)s
                """,
                url_rows,
            )
            updated = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    print(f"채용공고 URL {updated}행 백필 완료")
    return updated


if __name__ == "__main__":
    run()
