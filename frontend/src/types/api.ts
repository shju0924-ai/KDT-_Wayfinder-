// 백엔드 스키마와 1:1 대응 — backend/app/schemas/career.py
// 필드를 바꾸면 양쪽 모두 수정할 것.

// ── 경력 입력 ──────────────────────────────────────────
export interface CareerInput {
  raw_text: string;
  current_job_title?: string | null;
  /** STEP 1 진단 업무 — 업무에 붙은 NCS 코드를 STEP 2 능력단위 후보에 더한다(없으면 건너뜀) */
  diagnosed_tasks?: AutomationTask[];
}

export interface ParsedResume {
  filename: string;
  file_type: "pdf" | "docx" | "hwp" | "hwpx";
  text: string;
  char_count: number;
  warnings: string[];
}

/** 이력서가 없는 사용자를 위한 6문항 설문 응답 */
export interface SurveyInput {
  job_title: string;
  years: string;
  experience: string;
  strengths: string[];
  concern: string;
  aspiration?: string | null;
}

/** 설문 응답을 조립한 경력 서사 — 이후 STEP 2~4에 그대로 재사용한다 */
export interface SurveyCareer {
  career_text: string;
}

// ── STEP 1. 자동화 영향 진단 ───────────────────────────
/** 경력 서사에서 추출한 업무와 업무별 AI 영향 */
export interface AutomationTask {
  name: string;
  /** 직무에서 차지하는 비중 (0~100) */
  share_percent: number;
  automation_score: number;
  ai_assistance_score: number;
  effect: "automation" | "augmentation" | "human";
  rationale: string;
  ncs_code?: string | null;
  ncs_unit?: string | null;
  /** 이 업무의 근거가 된 경력 서사 원문 구절 */
  source_quote?: string | null;
}

/** 진단에 사용한 데이터 또는 방법론 출처 */
export interface RiskSource {
  source: string;
  label: string;
  url: string;
  /** AI 노출도 출처면 직업 간 순위로 환산한 0~100점(100에 가까울수록 AI로 대체될 가능성이 높음) */
  score?: number | null;
  note?: string | null;
}

/** 점수와 산출 근거만 담는다 — 낮음/보통/높음 같은 판정은 내지 않는다 */
export interface RiskDiagnosis {
  job_title: string;
  /** 자동화 위험도 (0~100) */
  risk_score: number;
  /** 점수 산출 방식 설명 */
  rationale: string;
  task_based_score: number;
  automation_share: number;
  augmentation_share: number;
  human_centered_share: number;
  /** 근거 충족도 (0~100) */
  confidence: number;
  tasks: AutomationTask[];
  sources: RiskSource[];
}

// ── STEP 2. 역량 프로필 ────────────────────────────────
/** 경력에서 확인한 역량 — NCS 능력단위(국가직무능력표준의 직무별 능력)로 표현한다 */
export interface SkillItem {
  /** NCS 능력단위 코드 (예: 0203020101_17v2) */
  ncs_code: string;
  /** 능력단위명 (예: 전표관리) */
  name: string;
  /** 능력단위가 속한 NCS 직무(세분류) (예: 회계·감사) */
  ncs_job: string;
  /** NCS 분류 경로 (대분류 > 중분류 > 소분류 > 세분류) */
  ncs_path: string;
  /** 능력단위 정의(축약) */
  definition: string;
  /** 이 능력단위의 근거가 된 경력 서사 원문 구절 — 사용자가 NCS 검색으로 추가하면 null */
  evidence?: string | null;
  /** 사용자가 직접 추가한 능력단위인지 — NCS 검색 또는 추천 후보에서 선택 */
  user_added: boolean;
  confirmed: boolean;
}

export interface SkillProfile {
  skills: SkillItem[];
  /** 검색 후보 중 AI가 고르지 않은 능력단위 — 사용자가 보고 골라 추가한다. evidence 는 후보를 찾은 경력 원문 구절 */
  suggestions?: SkillItem[];
  /** 현재(최근) 직무명 — 사용자 입력값(설문). 없으면 STEP 1 진단의 job_title 을 쓴다. STEP 3 같은 직무 제외에 사용 */
  current_job_title?: string | null;
}

// ── STEP 3. 인접 직무 탐색 ─────────────────────────────
export type JobSearchTrack = "adjacent_transition" | "training_transition";

export interface JobMatch {
  posting_id?: string | null;
  /** 같은 직무로 묶인 공고 수 */
  posting_count: number;
  job_title: string;
  company?: string | null;
  region?: "seoul" | "gg" | "incheon" | null;
  source_url?: string | null;
  /** 요구역량 판단 근거가 된 공고 발췌 */
  requirement_excerpt?: string | null;
  fit_score: number; // 0~100
  demand_outlook: "증가" | "유지" | "감소";
  transition_difficulty: "낮음" | "보통" | "높음";
  /** 이 공고와 연결되는 내 보유 역량 */
  matched_skills: string[];
  /** 공고에서 확인된 핵심 요구역량 */
  required_skills: string[];
  /** 요구역량 중 아직 없는 역량 */
  missing_skills: string[];
  /** 추천 근거가 약한 이유 — 비어 있으면 근거 충분 */
  weak_reasons: string[];
  /** 국가 면허가 있어야 일할 수 있는 직업인지 — 걸러내지 않고 라벨만 붙인다 */
  license_required: boolean;
  /** 이 직무의 AI 노출도 0~100 — 100에 가까울수록 AI로 대체될 가능성이 높은 직무.
   *  공개 자료(ILO·Anthropic)의 직업 간 순위를 출처별로 평균. 매칭 자료가 없으면 null */
  ai_exposure_score?: number | null;
  /** 노출도 출처별 점수와 매칭된 직업명 */
  ai_exposure_sources: RiskSource[];
}

// ── STEP 4. 학습 로드맵 ────────────────────────────────
export interface HrdCourse {
  name: string;
  institution?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  tuition?: number | null;
  address?: string | null;
  url?: string | null;
}

export interface RoadmapItem {
  skill_gap: string;
  learning_item: string;
  /** 예상 소요 기간(주). 0이면 훈련과정이 없어 기간 미정 */
  duration_weeks: number;
  resources: string[];
  source: string;
  course?: HrdCourse | null;
  /** 과정과 역량 격차의 연관 근거가 약한 이유 */
  weak_reason?: string | null;
}

export interface LearningRoadmap {
  target_job: string;
  items: RoadmapItem[];
}
