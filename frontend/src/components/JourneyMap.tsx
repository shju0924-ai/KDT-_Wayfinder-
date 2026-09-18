// 여정 지도 — "지금 어디에 있고, 다음에 무엇을 하면 되는지"를 한 화면에 담는다.
// 서비스의 중심 기능이라 채팅 상단에 고정되고, 히어로에서는 preview 형태로 미리 보여준다.
import type { ActionTarget, Journey, StageState } from "../lib/journey";
import { CheckIcon } from "./icons";

interface Props {
  journey: Journey;
  onAction: (target: ActionTarget) => void;
  /** 히어로 화면용 축약형 — 트랙만 보여주고 CTA·상세는 생략 */
  preview?: boolean;
}

const STATUS_TEXT: Record<StageState["status"], string> = {
  done: "완료",
  current: "진행 중",
  upcoming: "예정",
};

/** 달성률 도넛 — 숫자를 항상 가운데 함께 표기해 색에만 의존하지 않는다 */
function ProgressRing({ percent }: { percent: number }) {
  const r = 26;
  const c = 2 * Math.PI * r;
  return (
    <div className="jm-ring" role="img" aria-label={`전체 달성률 ${percent}퍼센트`}>
      <svg width="64" height="64" viewBox="0 0 64 64" aria-hidden>
        <circle cx="32" cy="32" r={r} fill="none" stroke="var(--line)" strokeWidth="6" />
        <circle
          className="jm-ring-fill"
          cx="32"
          cy="32"
          r={r}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - percent / 100)}
          transform="rotate(-90 32 32)"
        />
      </svg>
      <span className="jm-ring-num">
        {percent}
        <i>%</i>
      </span>
    </div>
  );
}

export default function JourneyMap({ journey, onAction, preview = false }: Props) {
  const { stages, current, percent, goal, nextAction, complete } = journey;

  return (
    <section className={`journey-map${preview ? " preview" : ""}`} aria-label="여정 진행 상황">
      {/* ── 헤더: 목표 + 달성률 ── */}
      <div className="jm-head">
        <div className="jm-head-text">
          <div className="jm-eyebrow">나의 여정</div>
          <h2 className="jm-goal">
            {goal ? (
              <>
                <span className="jm-goal-target">{goal}</span>
                <span className="jm-goal-suffix">(으)로 가는 길</span>
              </>
            ) : (
              "커리어 전환 여정"
            )}
          </h2>
        </div>
        <ProgressRing percent={percent} />
      </div>

      {/* ── 5단계 트랙 ── */}
      <ol className="jm-track" aria-label="여정 5단계">
        {stages.map((s) => (
          <li
            key={s.id}
            className={`jm-stage ${s.status}`}
            aria-current={s.status === "current" ? "step" : undefined}
          >
            <div className="jm-stage-rail" aria-hidden>
              <span className="jm-stage-bar" />
              <span className="jm-stage-node">
                {s.status === "done" ? <CheckIcon size={12} /> : s.step}
              </span>
            </div>
            <div className="jm-stage-text">
              <span className="jm-stage-label">{s.label}</span>
              <span className="jm-stage-status">{STATUS_TEXT[s.status]}</span>
            </div>
          </li>
        ))}
      </ol>

      {preview ? (
        <p className="jm-preview-note">
          {stages.map((s) => s.label).join(" · ")} — 다섯 단계를 함께 걸어요
        </p>
      ) : (
        /* ── 현재 단계 상세 + 다음 행동 ── */
        <div className="jm-now">
          <div className="jm-now-head">
            <span className="jm-now-badge">
              {complete ? "여정 완주" : `${current.step}단계 · ${current.label}`}
            </span>
            {current.detail && <span className="jm-now-detail">{current.detail}</span>}
          </div>
          <p className="jm-now-purpose">{current.purpose}</p>
          <p className="jm-now-value">{current.value}</p>

          {/* 실행 단계처럼 내부 진행이 있는 경우만 서브 진행률 표시 */}
          {current.id === "act" && current.detail && (
            <div className="jm-sub">
              <div
                className="jm-sub-bar"
                role="progressbar"
                aria-valuenow={Math.round(current.progress * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="학습 항목 완료율"
              >
                <span style={{ width: `${current.progress * 100}%` }} />
              </div>
              <span className="jm-sub-num">{Math.round(current.progress * 100)}%</span>
            </div>
          )}

          <button className="jm-cta" onClick={() => onAction(nextAction.target)}>
            <span className="jm-cta-label">{nextAction.label}</span>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
              <path
                d="M3 8h9m0 0L8.5 4.5M12 8l-3.5 3.5"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <p className="jm-cta-hint">{nextAction.hint}</p>
        </div>
      )}
    </section>
  );
}
