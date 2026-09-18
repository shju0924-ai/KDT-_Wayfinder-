// STEP 5 — 회고. 로드맵을 완주한 뒤 "무엇이 달라졌는지"를 확인하고 다음 방향을 정한다.
// 여정의 마지막 단계이자, 다음 여정의 시작점이 되는 화면.
import { CheckIcon } from "../icons";

interface Props {
  fromJob: string;
  toJob: string;
  skillCount: number;
  itemCount: number;
  totalWeeks: number;
  /** 회고 완료 처리 (여정 100% 달성) */
  onComplete: () => void;
  /** 이미 회고를 마쳤는지 */
  completed: boolean;
  onNewJourney: () => void;
}

export default function ReflectionCard({
  fromJob,
  toJob,
  skillCount,
  itemCount,
  totalWeeks,
  onComplete,
  completed,
  onNewJourney,
}: Props) {
  return (
    <div className="card reflect-card">
      <div className="card-eyebrow">STEP 5 · 회고</div>
      <h3>여정을 돌아봐요</h3>
      <p className="roadmap-summary">
        직무명 하나로만 설명되던 경력이, 이제 옮겨갈 수 있는 역량과 경로가 되었습니다.
      </p>

      <div className="reflect-route">
        <span className="rr-from">{fromJob}</span>
        <svg width="20" height="16" viewBox="0 0 20 16" fill="none" aria-hidden>
          <path
            d="M2 8h15m0 0-4.5-4.5M17 8l-4.5 4.5"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="rr-to">{toJob}</span>
      </div>

      <ul className="reflect-stats">
        <li>
          <strong>{skillCount}</strong>
          <span>재발견한 역량</span>
        </li>
        <li>
          <strong>{itemCount}</strong>
          <span>완주한 학습 항목</span>
        </li>
        <li>
          <strong>{totalWeeks}주</strong>
          <span>설계한 학습 기간</span>
        </li>
      </ul>

      <div className="card-foot">
        {completed ? (
          <span className="confirmed-note">
            <CheckIcon size={13} /> 여정을 완주했습니다
          </span>
        ) : (
          <button className="btn" onClick={onComplete}>
            회고 마치고 여정 완료하기
          </button>
        )}
        <button className="btn ghost" onClick={onNewJourney}>
          새 여정 시작하기
        </button>
      </div>
    </div>
  );
}
