// 응답 대기 표시 — 로컬 모델은 단계당 수 분~20분이라, 멈춘 것으로 오해하지 않도록 경과 시간을 보여준다.
import { useElapsedSeconds } from "../hooks/useElapsedSeconds";
import { CompassMark } from "./icons";

export default function BusyMessage() {
  const seconds = useElapsedSeconds(true);

  return (
    <div className="msg assistant">
      <div className="msg-avatar">
        <CompassMark size={18} />
      </div>
      <div className="msg-body">
        <span className="typing" aria-label="응답 작성 중">
          <i />
          <i />
          <i />
        </span>
        {seconds >= 10 && (
          <p className="busy-note">
            분석 중이에요 · {Math.floor(seconds / 60)}분 {seconds % 60}초 경과
            <br />
            로컬 AI 모델은 단계에 따라 최대 20분 정도 걸릴 수 있어요. 창을 닫지 말고 기다려주세요.
          </p>
        )}
      </div>
    </div>
  );
}
