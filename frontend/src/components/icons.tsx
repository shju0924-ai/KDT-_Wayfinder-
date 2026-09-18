// 공용 인라인 SVG 아이콘 — 외부 아이콘 라이브러리 없이 프로토타입 유지
interface IconProps {
  size?: number;
}

/** 나침반 로고 마크 — currentColor를 써서 다크 모드에서도 액센트 색을 그대로 상속 */
export function CompassMark({ size = 28 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" fill="none" aria-hidden style={{ color: "var(--accent)" }}>
      <circle cx="14" cy="14" r="12.5" stroke="currentColor" strokeWidth="2" />
      <path d="M18.5 9.5 15.6 15.6 9.5 18.5 12.4 12.4Z" fill="currentColor" />
      <circle cx="14" cy="14" r="1.6" fill="var(--surface)" />
    </svg>
  );
}

export function SunIcon({ size = 16 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" aria-hidden>
      <circle cx="8" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M8 1v1.6M8 13.4V15M15 8h-1.6M2.6 8H1M12.9 3.1l-1.13 1.13M4.23 11.67 3.1 12.8M12.9 12.9l-1.13-1.13M4.23 4.33 3.1 3.2"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function MoonIcon({ size = 16 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M13.5 9.5A6 6 0 1 1 6.5 2.5a5 5 0 0 0 7 7Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function SendIcon({ size = 18 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden>
      <path
        d="M10 16V4m0 0 -5 5m5-5 5 5"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function AttachmentIcon({ size = 19 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden>
      <path
        d="m7.1 10.4 5.1-5.1a3 3 0 0 1 4.2 4.2l-7 7a4.5 4.5 0 0 1-6.4-6.4l7.2-7.2"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function CheckIcon({ size = 14 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none" aria-hidden>
      <path
        d="m2.5 7.5 3 3 6-7"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function PlusIcon({ size = 16 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" aria-hidden>
      <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function XIcon({ size = 13 }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none" aria-hidden>
      <path d="m3 3 8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

/** 위험도 레벨 배지용 아이콘 — 색에만 의존하지 않도록 항상 라벨과 함께 사용 */
export function LevelIcon({ level }: { level: "낮음" | "보통" | "높음" }) {
  if (level === "낮음") return <CheckIcon size={12} />;
  if (level === "보통")
    return (
      <svg width={12} height={12} viewBox="0 0 12 12" fill="none" aria-hidden>
        <path d="M2 6h8" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      </svg>
    );
  return (
    <svg width={12} height={12} viewBox="0 0 12 12" fill="none" aria-hidden>
      <path
        d="M6 1.5 11 10.5H1L6 1.5Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}
