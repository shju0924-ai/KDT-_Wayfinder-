// API 클라이언트 — 모든 백엔드 호출은 이 파일을 통해서만 한다.
import axios from "axios";
import type {
  CareerInput,
  JobMatch,
  JobSearchTrack,
  LearningRoadmap,
  ParsedResume,
  SkillProfile,
  SurveyCareer,
  SurveyInput,
} from "../types/api";

// LLM 호출이 포함된 단계는 응답까지 수십 초가 걸릴 수 있어 타임아웃을 넉넉히 둔다.
const api = axios.create({ baseURL: "/api", timeout: 180_000 });

// 경력 입력 — 이력서 파일 파싱
export const parseResumeFile = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post<ParsedResume>("/diagnosis/parse-file", form).then((r) => r.data);
};

/** 설문 → 경력 서사 조립 (이력서 없는 사용자 경로) */
export const composeSurvey = (survey: SurveyInput) =>
  api.post<SurveyCareer>("/diagnosis/survey", survey).then((r) => r.data);

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

// STEP 4
export const generateRoadmap = (profile: SkillProfile, targetJob: string) =>
  api
    .post<LearningRoadmap>("/roadmap", { profile, target_job: targetJob })
    .then((r) => r.data);
