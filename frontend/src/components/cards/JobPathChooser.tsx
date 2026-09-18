import type { JobSearchTrack } from "../../types/api";

interface Props {
  selected: JobSearchTrack | null;
  disabled: boolean;
  onSelect: (track: JobSearchTrack) => void;
}

const paths: Array<{
  id: JobSearchTrack;
  eyebrow: string;
  title: string;
  description: string;
}> = [
  {
    id: "adjacent_transition",
    eyebrow: "경험을 이어서 전환",
    title: "인접 직무 전환형",
    description: "현재 경험과 역량은 활용하되, 같은 직무가 아닌 유사·인접 직무를 찾습니다.",
  },
  {
    id: "training_transition",
    eyebrow: "새 기술을 배워 전환",
    title: "교육 후 직무 전환형",
    description: "현재 일과 다른 분야에서 직업교육·자격·도구 학습 후 진입할 직무를 찾습니다.",
  },
];

export default function JobPathChooser({ selected, disabled, onSelect }: Props) {
  return (
    <div className="card">
      <div className="card-eyebrow">STEP 3 · 탐색 방향 선택</div>
      <h3>어떤 방식으로 다음 직무를 찾아볼까요?</h3>
      <p className="job-path-note">
        선택한 방향을 실제 채용공고 검색과 후보 선별에 함께 반영합니다.
      </p>
      <div className="job-path-grid">
        {paths.map((path) => {
          const isSelected = selected === path.id;
          return (
            <button
              type="button"
              className={`job-path-option${isSelected ? " selected" : ""}`}
              key={path.id}
              disabled={disabled || isSelected}
              aria-pressed={isSelected}
              onClick={() => onSelect(path.id)}
            >
              <span className="job-path-eyebrow">{path.eyebrow}</span>
              <strong>{path.title}</strong>
              <span>{path.description}</span>
              <b>{isSelected ? "선택됨" : "이 방향으로 공고 찾기 →"}</b>
            </button>
          );
        })}
      </div>
    </div>
  );
}
