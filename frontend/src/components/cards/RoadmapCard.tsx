// STEP 4 — 학습 로드맵. 보기만 하는 목록이 아니라 실제로 완주해 나가는 실행 트래커다.
// 체크한 항목은 여정의 '실행' 단계 진행률에 그대로 반영된다.
// RAG 근거 출처(source)는 항상 함께 표기해 신뢰성을 확보한다.
import type { LearningRoadmap } from "../../types/api";
import { CheckIcon } from "../icons";

const formatDate = (value?: string | null) => {
  if (!value) return null;
  const digits = value.replace(/\D/g, "");
  return digits.length === 8
    ? `${digits.slice(0, 4)}.${digits.slice(4, 6)}.${digits.slice(6, 8)}`
    : value;
};

interface Props {
  data: LearningRoadmap;
  /** 완료 처리된 학습 항목 이름들 */
  done: string[];
  onToggle: (learningItem: string) => void;
}

export default function RoadmapCard({ data, done, onToggle }: Props) {
  const totalWeeks = data.items.reduce((sum, it) => sum + it.duration_weeks, 0);
  const withCourse = data.items.filter((it) => it.course).length;
  const doneSet = new Set(done);
  const doneCount = data.items.filter((it) => doneSet.has(it.learning_item)).length;
  const total = data.items.length;
  const percent = total > 0 ? Math.round((doneCount / total) * 100) : 0;

  // 아직 완료되지 않은 첫 항목 = 지금 할 일
  const nextItem = data.items.find((it) => !doneSet.has(it.learning_item))?.learning_item ?? null;

  return (
    <div className="card">
      <div className="card-eyebrow">STEP 4 · 학습 로드맵</div>
      <h3>{data.target_job} 전환 로드맵</h3>
      <p className="roadmap-summary">
        총 {total}개 학습 항목 · 훈련과정 {withCourse}개 연결 · 약 {totalWeeks}주
        {total > withCourse && ` (훈련과정 없는 ${total - withCourse}개 항목은 기간 미정)`}
      </p>

      {/* 완주 진행률 — 실행 단계의 핵심 지표 */}
      <div className="roadmap-progress">
        <div
          className="rp-bar"
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="로드맵 완료율"
        >
          <span style={{ width: `${percent}%` }} />
        </div>
        <span className="rp-num">
          {doneCount}/{total} 완료
        </span>
      </div>

      {doneCount === total && total > 0 && (
        <p className="roadmap-cleared">
          <CheckIcon size={13} /> 모든 학습 항목을 완주했어요. 여정을 돌아볼 차례입니다.
        </p>
      )}

      <ol className="timeline">
        {data.items.map((it) => {
          const isDone = doneSet.has(it.learning_item);
          const isNext = it.learning_item === nextItem;
          const courseUrl =
            it.course?.url && /^https:\/\/(?:www\.)?work24\.go\.kr\//i.test(it.course.url)
              ? it.course.url
              : null;
          const coursePeriod = [formatDate(it.course?.start_date), formatDate(it.course?.end_date)]
            .filter(Boolean)
            .join(" ~ ");
          return (
            <li
              className={`tl-item${isDone ? " done" : ""}${isNext ? " next" : ""}`}
              key={it.learning_item}
            >
              <div className="tl-rail">
                <button
                  type="button"
                  className="tl-check"
                  role="checkbox"
                  aria-checked={isDone}
                  aria-label={`${it.learning_item} 완료 표시`}
                  onClick={() => onToggle(it.learning_item)}
                >
                  {isDone && <CheckIcon size={11} />}
                </button>
                <div className="tl-line" />
              </div>
              <div className="tl-body">
                <div className="tl-meta">
                  {/* 0주 = 맞는 훈련과정이 DB에 없어 기간을 알 수 없는 항목 */}
                  <span className="tl-weeks">
                    {it.duration_weeks > 0 ? `${it.duration_weeks}주` : "기간 미정"}
                  </span>
                  {!it.course && <span className="tag">훈련과정 없음</span>}
                  {isNext && <span className="tl-now">지금 할 일</span>}
                </div>
                <div className="tl-title">{it.learning_item}</div>
                <div className="tl-gap">보완 역량 — {it.skill_gap}</div>
                {it.weak_reason && (
                  <div className="weak-evidence" role="note">
                    <strong>근거 약함</strong> {it.weak_reason}
                  </div>
                )}
                {it.course && (
                  <div className="tl-course">
                    <span className="tl-course-label">고용24 실제 훈련과정</span>
                    <strong>{it.course.name}</strong>
                    {it.course.institution && <span>{it.course.institution}</span>}
                    <dl>
                      {coursePeriod && (
                        <div>
                          <dt>교육 기간</dt>
                          <dd>{coursePeriod}</dd>
                        </div>
                      )}
                      {it.course.tuition != null && (
                        <div>
                          <dt>훈련비</dt>
                          <dd>{it.course.tuition.toLocaleString("ko-KR")}원</dd>
                        </div>
                      )}
                      {it.course.address && (
                        <div>
                          <dt>교육 장소</dt>
                          <dd>{it.course.address}</dd>
                        </div>
                      )}
                    </dl>
                    {courseUrl && (
                      <a href={courseUrl} target="_blank" rel="noopener noreferrer">
                        고용24에서 과정 확인·신청하기 ↗
                      </a>
                    )}
                    <small>실제 모집 여부와 자비부담금은 고용24에서 최종 확인해주세요.</small>
                  </div>
                )}
                <ul className="tl-resources">
                  {it.resources.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
                <div className="tl-source">근거: {it.source}</div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
