// 여정(Journey) 도메인 — 5단계 정의와 상태 파생.
//
// 서비스의 3단계 파이프라인(역량 분해·직무 탐색·로드맵)은 사용자 입장에서 보면
// 시작 → 탐색 → 계획 까지다. 로드맵을 "받는" 것으로 끝나지 않고 실제로
// 실행하고 회고해야 여정이 완주된다 — 그래서 5단계로 표현한다.
//
// 이 파일은 순수 함수만 둔다(렌더링 의존 없음). Chat이 가진 상태를
// JourneyInput으로 넘기면 화면이 필요로 하는 모든 것을 계산해 돌려준다.

export type StageId = "start" | "explore" | "plan" | "act" | "reflect";
export type StageStatus = "done" | "current" | "upcoming";

/** CTA를 눌렀을 때 화면이 무엇을 해야 하는지 */
export type ActionTarget = "composer" | "skills" | "jobs" | "roadmap" | "reflect";

export interface StageDef {
  id: StageId;
  /** 1-based 순번 — 화면 표기용 */
  step: number;
  label: string;
  /** 이 단계에서 무엇을 하는지 */
  purpose: string;
  /** 이 단계를 지나면 사용자가 얻는 것 */
  value: string;
}

export const STAGES: readonly StageDef[] = [
  {
    id: "start",
    step: 1,
    label: "시작",
    purpose: "이번 여정의 목표를 정해요",
    value: "숨어 있던 강점이 드러납니다",
  },
  {
    id: "explore",
    step: 2,
    label: "탐색",
    purpose: "나에게 맞는 선택지와 방향을 살펴봐요",
    value: "갈 수 있는 길이 보입니다",
  },
  {
    id: "plan",
    step: 3,
    label: "계획",
    purpose: "현실적인 일정과 행동 계획을 만들어요",
    value: "막연함이 일정으로 바뀝니다",
  },
  {
    id: "act",
    step: 4,
    label: "실행",
    purpose: "작은 실천을 쌓아 목표에 가까워져요",
    value: "매일의 진전이 쌓입니다",
  },
  {
    id: "reflect",
    step: 5,
    label: "회고",
    purpose: "여정을 돌아보고 다음 방향을 발견해요",
    value: "다음 목표가 선명해집니다",
  },
] as const;

/** Chat이 들고 있는 원천 상태 */
export interface JourneyInput {
  hasCareer: boolean;
  hasSkillDraft: boolean;
  hasConfirmedSkills: boolean;
  hasJobs: boolean;
  selectedJob: string | null;
  hasRoadmap: boolean;
  totalItems: number;
  doneItems: number;
  /** 다음에 해야 할 학습 항목 이름 (실행 단계 CTA 힌트) */
  nextItemName: string | null;
  reflected: boolean;
}

export interface StageState extends StageDef {
  status: StageStatus;
  /** 단계 내부 진행률 0~1 — 실행 단계처럼 여러 항목이 있는 경우에 쓰인다 */
  progress: number;
  /** 트랙에 함께 보여줄 짧은 상태 문구 (예: "3/7 완료") */
  detail: string | null;
}

export interface NextAction {
  label: string;
  hint: string;
  target: ActionTarget;
}

export interface Journey {
  stages: StageState[];
  current: StageState;
  /** 전체 달성률 0~100 (정수) */
  percent: number;
  /** 목표 직무 — 아직 안 정해졌으면 null */
  goal: string | null;
  nextAction: NextAction;
  /** 모든 단계 완료 */
  complete: boolean;
}

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));

/**
 * 각 단계의 완료 여부와 내부 진행률을 계산한다.
 * done/progress만 여기서 정하고, current/upcoming은 아래에서 순서대로 결정한다.
 */
function measure(input: JourneyInput): Record<StageId, { done: boolean; progress: number; detail: string | null }> {
  const {
    hasCareer,
    hasSkillDraft,
    hasConfirmedSkills,
    selectedJob,
    hasRoadmap,
    totalItems,
    doneItems,
    reflected,
  } = input;

  // 탐색은 역량 확정(절반)과 직무 선택(나머지)으로 이뤄진다
  const exploreProgress = selectedJob ? 1 : hasConfirmedSkills ? 0.6 : hasSkillDraft ? 0.25 : 0;
  const actProgress = totalItems > 0 ? clamp01(doneItems / totalItems) : 0;

  return {
    start: {
      done: hasCareer,
      progress: hasCareer ? 1 : 0,
      detail: null,
    },
    explore: {
      done: selectedJob !== null,
      progress: exploreProgress,
      detail: selectedJob ? selectedJob : hasConfirmedSkills ? "직무 선택 대기" : null,
    },
    plan: {
      done: hasRoadmap,
      progress: hasRoadmap ? 1 : 0,
      detail: hasRoadmap ? `${totalItems}개 항목` : null,
    },
    act: {
      done: totalItems > 0 && doneItems >= totalItems,
      progress: actProgress,
      detail: totalItems > 0 ? `${doneItems}/${totalItems} 완료` : null,
    },
    reflect: {
      done: reflected,
      progress: reflected ? 1 : 0,
      detail: null,
    },
  };
}

function buildNextAction(current: StageState, input: JourneyInput): NextAction {
  switch (current.id) {
    case "start":
      return {
        label: "경력 입력하고 시작하기",
        hint: "이력서를 첨부하거나 해온 일을 그대로 적어주세요",
        target: "composer",
      };
    case "explore":
      if (!input.hasSkillDraft) {
        return {
          label: "역량 분해 요청하기",
          hint: "경력 속 전이 가능한 역량을 찾아드려요",
          target: "skills",
        };
      }
      if (!input.hasConfirmedSkills) {
        return {
          label: "역량 프로필 확정하기",
          hint: "AI가 찾은 역량을 검토하고 직접 고칠 수 있어요",
          target: "skills",
        };
      }
      return {
        label: "전환할 직무 선택하기",
        hint: "적합도와 함께 수요 전망·전환 난이도까지 살펴보세요",
        target: "jobs",
      };
    case "plan":
      return {
        label: "학습 로드맵 확인하기",
        hint: "목표 직무와의 역량 격차를 일정으로 바꿔드려요",
        target: "roadmap",
      };
    case "act":
      return {
        label: input.doneItems === 0 ? "첫 학습 항목 시작하기" : "다음 학습 항목 완료하기",
        hint: input.nextItemName ?? "로드맵에서 완료한 항목을 체크해주세요",
        target: "roadmap",
      };
    case "reflect":
      return {
        label: "여정 돌아보기",
        hint: "무엇이 달라졌는지 확인하고 다음 목표를 정해요",
        target: "reflect",
      };
  }
}

/** 앱 상태 → 화면이 필요로 하는 여정 정보 전체 */
export function deriveJourney(input: JourneyInput): Journey {
  const m = measure(input);

  // 앞에서부터 처음으로 완료되지 않은 단계가 '현재'
  const firstUndone = STAGES.findIndex((s) => !m[s.id].done);
  const currentIndex = firstUndone === -1 ? STAGES.length - 1 : firstUndone;
  const complete = firstUndone === -1;

  const stages: StageState[] = STAGES.map((def, i) => {
    const { done, progress, detail } = m[def.id];
    const status: StageStatus = done ? "done" : i === currentIndex ? "current" : "upcoming";
    return { ...def, status, progress, detail };
  });

  // 달성률 — 단계당 균등 배분하되 진행 중인 단계는 내부 진행률만큼만 인정
  const earned = stages.reduce(
    (sum, s) => sum + (s.status === "done" ? 1 : s.status === "current" ? s.progress : 0),
    0,
  );
  const percent = Math.round((earned / STAGES.length) * 100);

  const current = stages[currentIndex];

  return {
    stages,
    current,
    percent,
    goal: input.selectedJob,
    nextAction: buildNextAction(current, input),
    complete,
  };
}

/** 아직 아무것도 시작하지 않은 초기 여정 (히어로 화면 프리뷰용) */
export const EMPTY_JOURNEY_INPUT: JourneyInput = {
  hasCareer: false,
  hasSkillDraft: false,
  hasConfirmedSkills: false,
  hasJobs: false,
  selectedJob: null,
  hasRoadmap: false,
  totalItems: 0,
  doneItems: 0,
  nextItemName: null,
  reflected: false,
};
