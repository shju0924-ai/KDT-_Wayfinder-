// STEP 1 — 자동화 영향 진단 카드. 점수·산출 근거·출처만 보여준다.
// '위험 높음' 같은 판정 등급이나 경고색은 쓰지 않는다 — 숫자를 어떻게 받아들일지는 사용자가 정한다.
import type { AutomationTask, RiskDiagnosis } from "../../types/api";

const EFFECT_LABELS: Record<AutomationTask["effect"], string> = {
  automation: "AI가 대신할 수 있음",
  augmentation: "AI가 보조",
  human: "사람 중심",
};

interface Props {
  data: RiskDiagnosis;
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="fit-row">
      <div className="fit-bar" aria-hidden>
        <div className="fit-fill" style={{ width: `${value}%` }} />
      </div>
      <span className="fit-num">
        {label} {value}
      </span>
    </div>
  );
}

export default function RiskDiagnosisCard({ data }: Props) {
  return (
    <div className="card">
      <div className="card-eyebrow">STEP 1 · AI 자동화 영향 점수</div>
      <h3>{data.job_title} 업무의 AI 영향</h3>
      <p className="job-estimate-note">
        점수는 참고값입니다. 높고 낮음의 의미는 판단하지 않았어요 — 업무별 근거를 보고 직접 해석해 주세요.
      </p>

      <div className="risk-score">
        <span className="risk-score-num">{data.risk_score}</span>
        <span className="risk-score-unit">/ 100</span>
      </div>
      <p className="risk-formula">{data.rationale}</p>

      <div className="risk-shares" aria-label="업무 비중별 AI 영향">
        <div>
          <small>AI가 대신할 수 있는 업무</small>
          <strong>{data.automation_share}%</strong>
        </div>
        <div>
          <small>AI가 보조하는 업무</small>
          <strong>{data.augmentation_share}%</strong>
        </div>
        <div>
          <small>사람 중심 업무</small>
          <strong>{data.human_centered_share}%</strong>
        </div>
      </div>

      <ul className="risk-tasks">
        {data.tasks.map((t) => (
          <li key={t.name}>
            <div className="risk-task-head">
              <span className="risk-task-name">{t.name}</span>
              <span className="tag">업무 비중 {t.share_percent}%</span>
              <span className="tag">{EFFECT_LABELS[t.effect] ?? t.effect}</span>
            </div>
            <ScoreBar value={t.automation_score} label="자동화" />
            <ScoreBar value={t.ai_assistance_score} label="AI 보조" />
            <div className="risk-task-why">근거 — {t.rationale}</div>
            {t.ncs_unit && (
              <div className="risk-task-ncs">
                NCS 능력단위 {t.ncs_unit}
                {t.ncs_code && ` (${t.ncs_code})`}
              </div>
            )}
          </li>
        ))}
      </ul>

      <details className="risk-sources">
        <summary>
          출처 {data.sources.length}건 · 근거 충족도 {data.confidence}
        </summary>
        <ul>
          {data.sources.map((s) => (
            <li key={`${s.source}-${s.label}`}>
              <a href={s.url} target="_blank" rel="noopener noreferrer">
                {s.source}
              </a>{" "}
              — {s.label}
              {s.score != null && ` · ${s.score}`}
              {s.note && <small> ({s.note})</small>}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
