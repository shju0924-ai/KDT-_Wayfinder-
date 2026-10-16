from app.services.embedding import build_work24_course_url


def test_build_work24_course_url_from_hrd_source_id():
    url = build_work24_course_url("hrdnet:AIG20250000536706#2")

    assert url is not None
    assert "tracseId=AIG20250000536706" in url
    assert "tracseTme=2" in url
    assert url.startswith("https://www.work24.go.kr/")


def test_build_work24_course_url_ignores_chunk_suffix():
    url = build_work24_course_url("hrdnet:AIG20250000536706#2#c1")

    assert url is not None
    assert "tracseTme=2" in url


def test_build_work24_course_url_rejects_unknown_source():
    assert build_work24_course_url("manual:course-1") is None


def test_course_search_prefers_not_started_and_falls_back(monkeypatch):
    """개강 전 과정만 먼저 찾고, 하나도 없을 때만 끝나지 않은 과정(이미 개강)으로 물러선다."""
    import asyncio

    from app.services import embedding

    seen = []

    async def fake_embed(text):
        return [0.0]

    async def fake_run_db(fn, *args):
        if fn is embedding._execute_hnsw:
            seen.append(args[1]["min_start"])
            return [] if len(seen) == 1 else [{"source_id": "hrdnet:A#1", "ncs_cd": None}]
        return {}

    monkeypatch.setattr(embedding, "embed_query", fake_embed)
    monkeypatch.setattr(embedding, "run_db", fake_run_db)
    out = asyncio.run(embedding.search_similar_courses("조직관리", 3))
    assert seen[1] == "1900-01-01" and seen[0] != seen[1]
    assert out[0]["source_url"].startswith(embedding.WORK24_COURSE_DETAIL_URL)
