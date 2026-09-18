// 좌측 사이드바 — 브랜드, 새 여정, 5단계 여정 진행 상태(항상 보이는 레일)
import type { Journey } from "../lib/journey";
import { CheckIcon, CompassMark, MoonIcon, PlusIcon, SunIcon } from "./icons";

interface Props {
  journey: Journey;
  onNewChat: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export default function Sidebar({ journey, onNewChat, theme, onToggleTheme }: Props) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <CompassMark />
        <span className="brand-name">Wayfinder</span>
        <button
          className="theme-toggle"
          onClick={onToggleTheme}
          aria-label={theme === "dark" ? "라이트 모드로 전환" : "다크 모드로 전환"}
          title={theme === "dark" ? "라이트 모드로 전환" : "다크 모드로 전환"}
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
      <p className="brand-tagline">AI는 지도를 제공하고, 길은 인간이 선택한다</p>

      <button className="new-chat" onClick={onNewChat}>
        <PlusIcon /> 새 여정
      </button>

      <nav className="steps" aria-label="여정 진행 단계">
        <div className="steps-head">
          <span className="steps-label">나의 여정</span>
          <span className="steps-percent">{journey.percent}%</span>
        </div>
        <div className="steps-bar" aria-hidden>
          <span style={{ width: `${journey.percent}%` }} />
        </div>

        <ol className="steps-list">
          {journey.stages.map((s) => (
            <li
              className={`step ${s.status}`}
              key={s.id}
              aria-current={s.status === "current" ? "step" : undefined}
            >
              <span className="step-dot" aria-hidden>
                {s.status === "done" ? <CheckIcon size={11} /> : s.step}
              </span>
              <span className="step-text">
                <span className="step-label">{s.label}</span>
                {s.status === "current" && <span className="step-purpose">{s.purpose}</span>}
              </span>
            </li>
          ))}
        </ol>
      </nav>

      <div className="sidebar-foot">디자인 프로토타입 — 예시 데이터로 동작합니다</div>
    </aside>
  );
}
