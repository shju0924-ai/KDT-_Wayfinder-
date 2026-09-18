"""전체 데이터 파이프라인 오케스트레이터.

수집(서울·경기 채용공고 + HRD-Net 훈련과정) → 청킹 → 임베딩 → pgvector 적재를
한 번에 실행한다.

사용:
    python run_pipeline.py                    # 전체 실행 (기본 상한: 데모 규모)
    python run_pipeline.py --skip-collect     # 수집 생략, 기존 CSV로 임베딩·적재만
    python run_pipeline.py --limit 20         # 임베딩·적재 건수 제한 (소량 테스트)
    python run_pipeline.py --dry-run          # DB 적재 없이 임베딩까지만 검증

사전 조건:
    - data-pipeline/.env 에 UPSTAGE_API_KEY·SEOUL_JOB_API_KEY·GG_JOBA_API_KEY·
      HRDNET_API_KEY·DATABASE_URL 설정
    - DB 적재 시: docker compose up -d (프로젝트 루트) 로 PostgreSQL 실행 중이어야 함
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Wayfinder 데이터 파이프라인")
    parser.add_argument("--skip-collect", action="store_true", help="수집 단계 생략")
    parser.add_argument("--limit", type=int, default=None, help="임베딩·적재 건수 제한")
    parser.add_argument("--dry-run", action="store_true", help="DB 적재 없이 임베딩까지만")
    args = parser.parse_args()

    if not args.skip_collect:
        print("=" * 60)
        print("[1/2] 수집 단계")
        print("=" * 60)
        from collectors import automation_reference, gg_joba, hrdnet, ncs, seoul_job

        steps = [
            ("서울 일자리포털", seoul_job.collect),
            ("경기 잡아바", gg_joba.collect),
            ("HRD-Net 훈련과정", hrdnet.collect),
            ("NCS 능력단위", ncs.collect),
            ("자동화 위험도 참고자료", automation_reference.collect),
        ]
        for name, fn in steps:
            try:
                fn()
            except Exception as e:  # 한 소스 실패가 전체를 막지 않도록
                print(f"[경고] {name} 수집 실패: {e}", file=sys.stderr)

    print("=" * 60)
    print("[2/2] 청킹·임베딩·적재 단계")
    print("=" * 60)
    from embedding import embed_courses, embed_jobs, load_reference_data

    embed_jobs.run(limit=args.limit, dry_run=args.dry_run)
    embed_courses.run(limit=args.limit, dry_run=args.dry_run)
    load_reference_data.run(dry_run=args.dry_run)

    print("파이프라인 완료")


if __name__ == "__main__":
    main()
