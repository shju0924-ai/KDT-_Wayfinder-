// active 가 켜진 동안의 경과 초 — 켜질 때마다 0부터 다시 센다.
import { useEffect, useState } from "react";

export function useElapsedSeconds(active: boolean) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (!active) return;
    setSeconds(0);
    const started = Date.now();
    const id = window.setInterval(
      () => setSeconds(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(id);
  }, [active]);

  return seconds;
}
