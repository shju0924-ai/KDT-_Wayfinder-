// 백엔드 스키마와 1:1 대응 — backend/app/schemas/career.py
// 필드를 바꾸면 양쪽 모두 수정할 것.

// ── 경력 입력 ──────────────────────────────────────────
export interface CareerInput {
  raw_text: string;
  current_job_title?: string | null;
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

// ── STEP 2. 역량 프로필 ────────────────────────────────
export interface SkillItem {
  name: string;
  category: string;
  evidence: string;
  confirmed: boolean;
}

export interface SkillProfile {
  skills: SkillItem[];
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
  demand_outlook: string;
  transition_difficulty: "낮음" | "보통" | "높음";
  /** 이 공고와 연결되는 내 보유 역량 */
  matched_skills: string[];
  /** 공고에서 확인된 핵심 요구역량 */
  required_skills: string[];
  /** 요구역량 중 아직 없는 역량 */
  missing_skills: string[];
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
  duration_weeks: number;
  resources: string[];
  source: string;
  course?: HrdCourse | null;
}

export interface LearningRoadmap {
  target_job: string;
  items: RoadmapItem[];
}
