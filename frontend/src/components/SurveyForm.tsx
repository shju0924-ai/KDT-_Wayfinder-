// 6문항 경력 설문 — 이력서 없이도 시작할 수 있는 경로.
// 답변은 백엔드(services/survey.py)에서 경력 서사로 조립되므로, 여기서는
// 사용자가 쓴 문장을 가공하지 않고 그대로 전달한다.
import { useState } from "react";
import type { SurveyInput } from "../types/api";

interface Props {
  onSubmit: (survey: SurveyInput) => void;
  onCancel: () => void;
  busy?: boolean;
}

const YEAR_OPTIONS = ["1년 미만", "1~3년", "3년 이상", "5년 이상", "10년 이상"];

const STRENGTH_OPTIONS = [
  "고객 응대",
  "문제 해결",
  "문서·데이터 관리",
  "협업·조율",
  "운영 개선",
  "현장 대응",
];

const CONCERN_OPTIONS = [
  "현재 직무의 전망이 불안해요",
  "내 강점을 잘 모르겠어요",
  "다음 직무를 정하지 못했어요",
  "배워야 할 것이 막막해요",
];

export default function SurveyForm({ onSubmit, onCancel, busy }: Props) {
  const [jobTitle, setJobTitle] = useState("");
  const [years, setYears] = useState(YEAR_OPTIONS[2]);
  const [experience, setExperience] = useState("");
  const [strengths, setStrengths] = useState<string[]>([]);
  const [concern, setConcern] = useState("");
  const [aspiration, setAspiration] = useState("");

  const toggleStrength = (s: string) =>
    setStrengths((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));

  // 필수 4개(직무·기간·경험·고민) + 강점 1개 이상
  const filled =
    [jobTitle.trim(), experience.trim(), concern].filter(Boolean).length +
    (strengths.length > 0 ? 1 : 0) +
    1; // 기간은 기본값이 있어 항상 채워짐
  const total = 5;
  const canSubmit =
    jobTitle.trim() !== "" && experience.trim() !== "" && concern !== "" && strengths.length > 0;

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit || busy) return;
    onSubmit({
      job_title: jobTitle.trim(),
      years,
      experience: experience.trim(),
      strengths,
      concern,
      aspiration: aspiration.trim() || null,
    });
  };

  return (
    <form className="survey" onSubmit={submit}>
      <header className="sv-head">
        <div className="sv-head-text">
          <div className="card-eyebrow">경력 이야기</div>
          <h2>나의 경력 이야기</h2>
          <p className="sv-sub">짧게 적어도 괜찮아요. 알고 있는 만큼만 답해주세요.</p>
        </div>
        <span className="sv-count">
          {Math.min(filled, total)} / {total} 작성
        </span>
      </header>

      <div
        className="sv-progress"
        role="progressbar"
        aria-valuenow={Math.min(filled, total)}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-label="설문 작성 진행률"
      >
        <span style={{ width: `${(Math.min(filled, total) / total) * 100}%` }} />
      </div>

      <div className="sv-grid">
        {/* Q1 · Q2 — 한 줄에 나란히 */}
        <div className="sv-field">
          <label className="sv-label" htmlFor="sv-job">
            <span className="sv-num">1</span> 현재 또는 가장 최근의 직무는 무엇인가요?
            <em className="sv-req">필수</em>
          </label>
          <input
            id="sv-job"
            className="sv-input"
            placeholder="예: 서비스 운영 매니저"
            value={jobTitle}
            onChange={(e) => setJobTitle(e.target.value)}
            required
          />
        </div>

        <div className="sv-field">
          <label className="sv-label" htmlFor="sv-years">
            <span className="sv-num">2</span> 이 일을 얼마나 하셨나요?
          </label>
          <select
            id="sv-years"
            className="sv-input"
            value={years}
            onChange={(e) => setYears(e.target.value)}
          >
            {YEAR_OPTIONS.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>

        {/* Q3 */}
        <div className="sv-field wide">
          <label className="sv-label" htmlFor="sv-exp">
            <span className="sv-num">3</span> 가장 자신 있게 설명할 수 있는 경험을 들려주세요.
            <em className="sv-req">필수</em>
          </label>
          <textarea
            id="sv-exp"
            className="sv-input sv-textarea"
            rows={5}
            placeholder="어떤 상황에서, 무슨 일을 했고, 결과가 어땠는지 함께 적어주세요."
            value={experience}
            onChange={(e) => setExperience(e.target.value)}
            required
          />
          <p className="sv-hint">성과의 크기보다 직접 맡았던 역할과 판단을 중심으로 적어주세요.</p>
        </div>

        {/* Q4 */}
        <fieldset className="sv-field wide">
          <legend className="sv-label">
            <span className="sv-num">4</span> 내가 자주 맡았거나 잘했던 일은 무엇인가요?
            <em className="sv-req">필수</em>
          </legend>
          <div className="sv-chips">
            {STRENGTH_OPTIONS.map((s) => {
              const on = strengths.includes(s);
              return (
                <button
                  type="button"
                  key={s}
                  className={`sv-chip${on ? " on" : ""}`}
                  aria-pressed={on}
                  onClick={() => toggleStrength(s)}
                >
                  {s}
                </button>
              );
            })}
          </div>
        </fieldset>

        {/* Q5 */}
        <fieldset className="sv-field wide">
          <legend className="sv-label">
            <span className="sv-num">5</span> 지금 커리어에서 가장 고민되는 점은 무엇인가요?
            <em className="sv-req">필수</em>
          </legend>
          <div className="sv-radios">
            {CONCERN_OPTIONS.map((c) => (
              <label key={c} className={`sv-radio${concern === c ? " on" : ""}`}>
                <input
                  type="radio"
                  name="concern"
                  value={c}
                  checked={concern === c}
                  onChange={() => setConcern(c)}
                />
                <span className="sv-radio-dot" aria-hidden />
                {c}
              </label>
            ))}
          </div>
        </fieldset>

        {/* Q6 */}
        <div className="sv-field wide">
          <label className="sv-label" htmlFor="sv-asp">
            <span className="sv-num">6</span> 앞으로 해보고 싶은 일이 있나요?
            <em className="sv-opt">선택</em>
          </label>
          <textarea
            id="sv-asp"
            className="sv-input sv-textarea"
            rows={3}
            placeholder="아직 잘 모르겠다면 비워두어도 괜찮아요."
            value={aspiration}
            onChange={(e) => setAspiration(e.target.value)}
          />
        </div>
      </div>

      <div className="sv-foot">
        <button type="button" className="btn ghost" onClick={onCancel} disabled={busy}>
          뒤로
        </button>
        <button type="submit" className="btn" disabled={!canSubmit || busy}>
          {busy ? "정리하는 중…" : "답변 제출하고 시작하기"}
        </button>
      </div>
    </form>
  );
}
