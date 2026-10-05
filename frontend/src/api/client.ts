// API 클라이언트 — 모든 백엔드 호출은 이 파일을 통해서만 한다.
import axios from "axios";
import type {
  CareerInput,
  JobMatch,
  JobSearchTrack,
  LearningRoadmap,
  ParsedResume,
  RiskDiagnosis,
  SkillProfile,
  SurveyCareer,
  SurveyInput,
} from "../types/api";

// LLM 호출이 포함된 단계는 응답까지 오래 걸린다 — 로컬 모델(Ollama, CPU)은 단계당 수 분~20분.
// 10분이던 값은 직무 매칭(실측 14~19분)보다 짧아 화면만 실패로 끝났다. 백엔드 OLLAMA_TIMEOUT_SECONDS 와 맞춘다.
const api = axios.create({ baseURL: "/api", timeout: 1_800_000 });

// 경력 입력 — 이력서 파일 파싱
export const parseResumeFile = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post<ParsedResume>("/diagnosis/parse-file", form).then((r) => r.data);
};

/** 설문 → 경력 서사 조립 (이력서 없는 사용자 경로) */
export const composeSurvey = (survey: SurveyInput) =>
  api.post<SurveyCareer>("/diagnosis/survey", survey).then((r) => r.data);

// STEP 1 — 업무별 AI 자동화 영향 점수 (판정 없이 점수·근거만)
export const diagnoseRisk = (input: CareerInput) =>
  api.post<RiskDiagnosis>("/diagnosis/risk", input).then((r) => r.data);

// STEP 2
export const generateProfile = (input: CareerInput) =>
  api.post<SkillProfile>("/profile", input).then((r) => r.data);

// STEP 3 — 선택한 탐색 경로에 맞춰 공고를 찾고 현재 직무는 제외한다
export const matchJobs = (
  profile: SkillProfile,
  searchTrack: JobSearchTrack,
  excludeJob?: string,
) =>
  api
    .post<JobMatch[]>("/jobs/match", profile, {
      params: {
        search_track: searchTrack,
        ...(excludeJob ? { exclude_job: excludeJob } : {}),
      },
    })
    .then((r) => r.data);

// STEP 4 — STEP 3 카드의 '새로 배워야 할 역량'을 넘기면 서버가 격차 추출 호출을 건너뛴다
export const generateRoadmap = (
  profile: SkillProfile,
  targetJob: string,
  missingSkills: string[] = [],
) =>
  api
    .post<LearningRoadmap>("/roadmap", {
      profile,
      target_job: targetJob,
      missing_skills: missingSkills,
    })
    .then((r) => r.data);
