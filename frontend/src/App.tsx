// 앱 셸 — 사이드바(여정 진행) + 챗 화면.
// 5단계 여정(시작·탐색·계획·실행·회고)은 별도 페이지가 아니라 대화 속 카드로 진행되고,
// 진행 상태는 사이드바 레일과 채팅 상단의 여정 지도에 함께 반영된다.
import { useCallback, useState } from "react";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import { useTheme } from "./hooks/useTheme";
import type { Journey } from "./lib/journey";
import { deriveJourney, EMPTY_JOURNEY_INPUT } from "./lib/journey";

export default function App() {
  const [journey, setJourney] = useState<Journey>(() => deriveJourney(EMPTY_JOURNEY_INPUT));
  const [sessionKey, setSessionKey] = useState(0);
  const { theme, toggle } = useTheme();

  const newChat = useCallback(() => {
    setJourney(deriveJourney(EMPTY_JOURNEY_INPUT));
    setSessionKey((k) => k + 1); // key 변경으로 Chat 상태 전체 리셋
  }, []);

  return (
    <div className="shell">
      <Sidebar journey={journey} onNewChat={newChat} theme={theme} onToggleTheme={toggle} />
      <Chat key={sessionKey} onJourney={setJourney} onNewChat={newChat} />
    </div>
  );
}
