// 대화 화면 상단의 여정 요약 — 결과를 읽는 흐름을 방해하지 않도록 요약으로 시작하고 필요할 때 펼친다.
import { useState } from "react";
import type { ActionTarget, Journey } from "../lib/journey";
import JourneyMap from "./JourneyMap";

interface Props {
  journey: Journey;
  onAction: (target: ActionTarget) => void;
}

export default function JourneyDock({ journey, onAction }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`journey-dock${expanded ? " expanded" : ""}`}>
      <button
        type="button"
        className="journey-toggle"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-controls="journey-detail"
      >
        <span className="journey-toggle-context">
          <span className="journey-toggle-eyebrow">나의 여정</span>
          <span className="journey-toggle-stage">
            {journey.complete ? "여정 완주" : `${journey.current.step}단계 · ${journey.current.label}`}
          </span>
        </span>
        <span className="journey-toggle-progress" aria-label={`전체 달성률 ${journey.percent}%`}>
          <span className="journey-toggle-bar" aria-hidden>
            <span style={{ width: `${journey.percent}%` }} />
          </span>
          <b>{journey.percent}%</b>
        </span>
        <span className="journey-toggle-action">
          {expanded ? "접기" : "전체 여정 보기"}
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
            <path
              d="m4 6 4 4 4-4"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>
      {expanded && (
        <div id="journey-detail">
          <JourneyMap journey={journey} onAction={onAction} />
        </div>
      )}
    </div>
  );
}
