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
