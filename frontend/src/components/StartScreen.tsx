// 시작 화면 — 방식 선택(설문 / 이력서 파일) 히어로와 설문 작성 화면.
// 시작 화면의 파일 선택은 자유 입력 Composer와 독립적으로 동작한다.
import { useRef, useState } from "react";
import type { SurveyInput } from "../types/api";
import type { ActionTarget, Journey } from "../lib/journey";
import JourneyMap from "./JourneyMap";
import StartChooser from "./StartChooser";
import SurveyForm from "./SurveyForm";
import { CompassMark } from "./icons";

interface Props {
  journey: Journey;
  busy: boolean;
  onFile: (file: File) => void;
  onSurveySubmit: (survey: SurveyInput) => void;
  onJourneyAction: (target: ActionTarget) => void;
}

export default function StartScreen({ journey, busy, onFile, onSurveySubmit, onJourneyAction }: Props) {
  const [mode, setMode] = useState<"choose" | "survey">("choose");
  const fileRef = useRef<HTMLInputElement>(null);

  if (mode === "survey") {
    return (
      <main className="chat">
        <div className="survey-view">
          <div className="survey-inner">
            <SurveyForm onSubmit={onSurveySubmit} onCancel={() => setMode("choose")} busy={busy} />
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="chat">
      <div className="hero">
        <div className="hero-inner">
          <CompassMark size={48} />
          <h1>어떤 방식으로 시작할까요?</h1>
          <p className="hero-sub">
            경력 속에 숨어 있는 역량을 찾아, 다음 커리어로 가는 지도를 함께 그려드릴게요.
            이력서가 없어도 괜찮아요.
          </p>

          <input
            ref={fileRef}
            className="file-input"
            type="file"
            accept=".pdf,.docx,.hwp,.hwpx"
            disabled={busy}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onFile(file);
              event.target.value = "";
            }}
          />

          <StartChooser
            onSurvey={() => setMode("survey")}
            onFile={() => fileRef.current?.click()}
            disabled={busy}
          />

          {/* 시작 전에도 앞으로의 여정을 미리 보여준다 */}
          <JourneyMap journey={journey} onAction={onJourneyAction} preview />

          <p className="hero-disclaimer">
            AI가 판정하지 않습니다 — 모든 단계는 당신이 검토하고 선택합니다.
          </p>
        </div>
      </div>
    </main>
  );
}
