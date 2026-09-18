// STEP 3 — 인접 직무 탐색 카드. 적합도·수요 전망·전환 난이도 비교 후 목표 직무 선택.
import type { JobMatch, JobSearchTrack } from "../../types/api";

interface Props {
  jobs: JobMatch[];
  searchTrack: JobSearchTrack;
  selected: string | null;
  onSelect: (job: JobMatch) => void;
}

export default function JobMatchList({ jobs, searchTrack, selected, onSelect }: Props) {
  const openPosting = (url: string) => {
    window.open(url, "_blank", "noopener,noreferrer");
  };

  return (
    <div className="card">
      <div className="card-eyebrow">
        STEP 3 · {searchTrack === "adjacent_transition" ? "인접 직무 전환형" : "교육 후 직무 전환형"}
      </div>
      <h3>
        {searchTrack === "adjacent_transition"
          ? "현재 경험이 이어지는 인접 후보 직무"
          : "새로 배워 전환할 수 있는 후보 직무"}
      </h3>
      <p className="job-estimate-note">
        수요 전망과 전환 난이도는 채용공고 및 보유 역량을 바탕으로 AI가 추정한 참고값입니다.
      </p>
      <div className="job-list">
        {jobs.map((job) => {
          const isSelected = selected === job.job_title;
          const dimmed = selected !== null && !isSelected;
          const postingUrl =
            job.source_url && /^https?:\/\//i.test(job.source_url) ? job.source_url : null;
          const hasPostingUrl = Boolean(postingUrl);
          return (
            <div
              key={job.job_title}
              className={`job-card${isSelected ? " selected" : ""}${dimmed ? " dimmed" : ""}${hasPostingUrl ? " clickable" : ""}`}
              role={hasPostingUrl ? "link" : undefined}
              tabIndex={hasPostingUrl ? 0 : undefined}
              aria-label={hasPostingUrl ? `${job.job_title} 채용공고 새 창에서 보기` : undefined}
              onClick={() => postingUrl && openPosting(postingUrl)}
              onKeyDown={(event) => {
                if (
                  event.target === event.currentTarget &&
                  postingUrl &&
                  (event.key === "Enter" || event.key === " ")
                ) {
                  event.preventDefault();
                  openPosting(postingUrl);
                }
              }}
            >
              <div className="job-head">
                <span className="job-title">{job.job_title}</span>
                <div className="job-meta">
                  <span className="tag" title="채용공고와 일반 직종 정보를 바탕으로 한 AI 추정">
                    AI 추정 수요 전망 {job.demand_outlook}
                  </span>
                  <span className="tag" title="보유 역량과 공고 요구사항의 거리를 바탕으로 한 AI 추정">
                    AI 추정 전환 난이도 {job.transition_difficulty}
                  </span>
                </div>
              </div>
              {(job.company || job.region) && (
                <div className="job-source">
                  실제 공고 · {job.company ?? "회사명 미기재"}
                  {job.region && ` · ${job.region === "seoul" ? "서울" : "경기"}`}
                  {job.posting_count > 1 && ` · 관련 공고 ${job.posting_count}건`}
                  {hasPostingUrl && " · 공고 보기 ↗"}
                </div>
              )}
              {searchTrack === "adjacent_transition" && (
                <div className="fit-row">
                  <div className="fit-bar" aria-hidden>
                    <div className="fit-fill" style={{ width: `${job.fit_score}%` }} />
                  </div>
                  <span className="fit-num">적합도 {job.fit_score}</span>
                </div>
              )}

              {/* 공고를 '내 역량 / 공고 요구 / 아직 없는 것' 세 축으로 나눠 보여준다 */}
              <div className="job-skill-groups">
                <div className="job-skill-group">
                  <small>연결되는 내 역량</small>
                  <div className="job-skills">
                    {job.matched_skills.length > 0 ? (
                      job.matched_skills.map((s) => (
                        <span className="tag match" key={s}>
                          {s}
                        </span>
                      ))
                    ) : (
                      <span className="job-skill-empty">추가 분석 필요</span>
                    )}
                  </div>
                </div>
                <div className="job-skill-group">
                  <small>공고 핵심 요구</small>
                  <div className="job-skills">
                    {job.required_skills.length > 0 ? (
                      job.required_skills.map((s) => (
                        <span className="tag requirement" key={s}>
                          {s}
                        </span>
                      ))
                    ) : (
                      <span className="job-skill-empty">공고 원문에서 확인</span>
                    )}
                  </div>
                </div>
                {job.missing_skills.length > 0 && (
                  <div className="job-skill-group">
                    <small>아직 없는 역량 — 로드맵에서 채웁니다</small>
                    <div className="job-skills">
                      {job.missing_skills.map((s) => (
                        <span className="tag gap" key={s}>
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              {!selected && (
                <div className="job-foot">
                  <button
                    className="btn ghost"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(job);
                    }}
                  >
                    이 직무로 로드맵 만들기 →
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
