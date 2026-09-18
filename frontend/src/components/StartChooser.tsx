// 시작 방식 선택 — 이력서가 있는 사람과 없는 사람 모두 STEP 1로 들어올 수 있게 한다.
// 어느 경로로 들어와도 결국 같은 '경력 서사'가 되어 STEP 2~4는 동일하게 진행된다.
import { AttachmentIcon, CheckIcon } from "./icons";

interface Props {
  onSurvey: () => void;
  onFile: () => void;
  disabled?: boolean;
}

const SURVEY_POINTS = [
  "이력서가 없어도 가능해요",
  "6개 질문에 답변합니다",
  "답변을 경력 서사로 정리해요",
];

const FILE_POINTS = [
  "PDF · DOCX · HWP를 지원해요",
  "경력 내용을 자동으로 추출해요",
  "부족한 내용은 대화로 보완해요",
];

function Points({ items }: { items: string[] }) {
  return (
    <ul className="sc-points">
      {items.map((t) => (
        <li key={t}>
          <CheckIcon size={12} />
          {t}
        </li>
      ))}
    </ul>
  );
}

export default function StartChooser({ onSurvey, onFile, disabled }: Props) {
  return (
    <section className="start-chooser" aria-label="시작 방식 선택">
      <div className="sc-grid">
        {/* 설문 경로 */}
        <div className="sc-card">
          <div className="sc-icon" aria-hidden>
            Q
          </div>
          <span className="sc-time">약 5분</span>
          <h3 className="sc-title">설문으로 시작하기</h3>
          <Points items={SURVEY_POINTS} />
          <button className="btn ghost sc-btn" onClick={onSurvey} disabled={disabled}>
            설문 시작
          </button>
        </div>

        {/* 파일 경로 — 이력서가 있으면 가장 빠른 길이라 기본 강조 */}
        <div className="sc-card primary">
          <div className="sc-icon" aria-hidden>
            <AttachmentIcon size={20} />
          </div>
          <span className="sc-time">약 2분</span>
          <h3 className="sc-title">파일로 빠르게 분석하기</h3>
          <Points items={FILE_POINTS} />
          <button className="btn sc-btn" onClick={onFile} disabled={disabled}>
            이력서 파일 선택
          </button>
        </div>
      </div>

      <p className="sc-note">
        두 방식 모두 결과를 직접 확인하고 수정할 수 있어요 — 입력한 정보는 역량 분석에만
        사용됩니다.
      </p>
    </section>
  );
}
