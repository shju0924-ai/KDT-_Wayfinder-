// 챗 화면 — 대화가 곧 여정이다.
// 역량 분해→직무 탐색→로드맵이 대화 속 인터랙티브 카드로 진행되며, 각 단계는 실제 백엔드 API를 호출한다.
// 로드맵을 받는 것으로 끝나지 않고, 항목을 하나씩 완료(실행)하고 돌아보는(회고) 데까지 이어진다.
// 각 단계의 확정(역량 확정, 직무 선택, 항목 완료)은 반드시 사용자의 행동으로만 넘어간다.
import { useEffect, useMemo, useRef, useState } from "react";
import {
  composeSurvey,
  diagnoseRisk,
  generateProfile,
  generateRoadmap,
  matchJobs,
  parseResumeFile,
} from "../api/client";
import type {
  CareerInput,
  JobMatch,
  JobSearchTrack,
  LearningRoadmap,
  RiskDiagnosis,
  SkillItem,
  SurveyInput,
} from "../types/api";
import type { ActionTarget, Journey, JourneyInput } from "../lib/journey";
import { deriveJourney } from "../lib/journey";
import Composer from "./Composer";
import JourneyMap from "./JourneyMap";
import StartChooser from "./StartChooser";
import SurveyForm from "./SurveyForm";
import { CompassMark } from "./icons";
import SkillProfileCard from "./cards/SkillProfileCard";
import JobMatchList from "./cards/JobMatchList";
import JobPathChooser from "./cards/JobPathChooser";
import RoadmapCard from "./cards/RoadmapCard";
import ReflectionCard from "./cards/ReflectionCard";
import RiskDiagnosisCard from "./cards/RiskDiagnosisCard";

type CardKind = "risk" | "skills" | "job-path" | "jobs" | "roadmap" | "reflect";

interface Msg {
  id: number;
  role: "user" | "assistant";
  text?: string;
  card?: CardKind;
  /** 빠른 응답 액션 — 역량 분해 시작 트리거 */
  action?: string;
  error?: boolean;
}

interface Props {
  onJourney: (journey: Journey) => void;
  onNewChat: () => void;
}

/** 사용자에게 보여줄 오류 메시지로 변환 (원인을 숨기지 않되 읽을 수 있게) */
function errorText(e: unknown): string {
  const status = (e as { response?: { status?: number } })?.response?.status;
  const detail =
    (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
    (e as Error)?.message ??
    String(e);
  if (status && status < 500) return `파일을 읽지 못했어요.\n(${detail})`;
  return `요청을 처리하지 못했어요.\n(${detail})\n\n백엔드가 실행 중인지, API 키가 설정돼 있는지 확인해주세요.`;
}

export default function Chat({ onJourney, onNewChat }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  // 로컬 모델은 단계당 수 분~20분 — 멈춘 것으로 오해하지 않도록 경과 시간을 보여준다
  const [busySeconds, setBusySeconds] = useState(0);
  useEffect(() => {
    if (!busy) return;
    setBusySeconds(0);
    const started = Date.now();
    const id = window.setInterval(
      () => setBusySeconds(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(id);
  }, [busy]);

  // 각 단계의 실제 API 응답
  const [career, setCareer] = useState<CareerInput | null>(null);
  const [risk, setRisk] = useState<RiskDiagnosis | null>(null);
  const [draftSkills, setDraftSkills] = useState<SkillItem[] | null>(null);
  const [confirmedSkills, setConfirmedSkills] = useState<SkillItem[] | null>(null);
  const [jobSearchTrack, setJobSearchTrack] = useState<JobSearchTrack | null>(null);
  const [jobs, setJobs] = useState<JobMatch[] | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobMatch | null>(null);
  const [roadmap, setRoadmap] = useState<LearningRoadmap | null>(null);

  // 실행·회고 단계 — 로드맵을 받은 뒤에도 여정은 계속된다
  const [doneItems, setDoneItems] = useState<string[]>([]);
  const [reflected, setReflected] = useState(false);
  const [reflectShown, setReflectShown] = useState(false);
  const [journeyExpanded, setJourneyExpanded] = useState(false);

  // 시작 화면 모드 — 방식 선택 → 설문 / 파일 / 자유 입력
  const [startMode, setStartMode] = useState<"choose" | "survey">("choose");

  const idRef = useRef(0);
  const endRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const dockRef = useRef<HTMLDivElement>(null);
  const startFileRef = useRef<HTMLInputElement>(null);

  const nextId = () => ++idRef.current;
  const push = (...items: Omit<Msg, "id">[]) =>
    setMessages((m) => [...m, ...items.map((it) => ({ ...it, id: nextId() }))]);

  // ── 여정 상태 파생 ───────────────────────────────────────
  const journey = useMemo<Journey>(() => {
    const input: JourneyInput = {
      hasCareer: career !== null,
      hasSkillDraft: draftSkills !== null,
      hasConfirmedSkills: confirmedSkills !== null,
      hasJobs: jobs !== null,
      selectedJob: selectedJob?.job_title ?? null,
      hasRoadmap: roadmap !== null,
      totalItems: roadmap?.items.length ?? 0,
      doneItems: doneItems.length,
      nextItemName:
        roadmap?.items.find((it) => !doneItems.includes(it.learning_item))?.learning_item ?? null,
      reflected,
    };
    return deriveJourney(input);
  }, [career, draftSkills, confirmedSkills, jobs, selectedJob, roadmap, doneItems, reflected]);

  useEffect(() => {
    onJourney(journey);
  }, [journey, onJourney]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  // 모든 학습 항목을 완료하면 회고 단계로 넘어간다 (한 번만)
  useEffect(() => {
    if (!roadmap || reflectShown) return;
    if (roadmap.items.length === 0 || doneItems.length < roadmap.items.length) return;
    setReflectShown(true);
    push(
      {
        role: "assistant",
        text: "로드맵의 모든 항목을 완료하셨네요. 여정을 한 번 돌아볼까요?",
      },
      { role: "assistant", card: "reflect" },
    );
  }, [doneItems, roadmap, reflectShown]);

  // 경력 서사를 확보하면 STEP 1(업무별 AI 영향 점수)을 보여주고 STEP 2(역량 분해)로 안내한다.
  // 진단은 참고 정보라 실패해도 여정은 막지 않는다.
  const startProfile = async (input: CareerInput) => {
    setCareer(input);
    setBusy(true);
    try {
      const result = await diagnoseRisk(input);
      setRisk(result);
      push(
        {
          role: "assistant",
          text: "먼저 지금 하는 업무들이 AI의 영향을 얼마나 받는지 점수로 나눠봤어요.\n좋다·나쁘다 판정은 하지 않았어요 — 업무별 근거를 보고 직접 판단해 주세요.",
        },
        { role: "assistant", card: "risk" },
      );
    } catch (e) {
      push({
        role: "assistant",
        text: `AI 영향 점수는 계산하지 못했어요. 역량 분석은 그대로 진행할 수 있어요.\n(${errorText(e)})`,
        error: true,
      });
    } finally {
      setBusy(false);
    }
    push({
      role: "assistant",
      text: "지금까지의 경험을 다른 직무로도 옮겨갈 수 있는 역량 단위로 분해해 볼까요?",
      action: "네, 제 역량을 분해해주세요",
    });
  };

  // ── 경력 입력 (자유 텍스트) → STEP 2 안내 ────────────────
  const handleSend = async (text: string) => {
    push({ role: "user", text });

    if (career) {
      // 파이프라인 진행 중 자유 입력 — 카드 버튼으로 진행하도록 안내
      push({
        role: "assistant",
        text: reflected
          ? "이번 여정을 완주하셨어요! 새로운 경력으로 다시 시작하려면 왼쪽의 '새 여정'을 눌러주세요."
          : "위 카드의 버튼으로 다음 단계를 진행해주세요 🙂",
      });
      return;
    }

    push({
      role: "assistant",
      text: "들려주신 경력을 경력 서사로 정리했어요.",
    });
    await startProfile({ raw_text: text, current_job_title: null });
  };

  const handleFile = async (file: File) => {
    push({ role: "user", text: `이력서 파일: ${file.name}` });
    if (career) {
      push({
        role: "assistant",
        text: reflected
          ? "이번 여정을 완주하셨어요! 새 파일로 시작하려면 왼쪽의 '새 여정'을 눌러주세요."
          : "현재 여정이 진행 중이에요. 새 파일로 시작하려면 왼쪽의 '새 여정'을 눌러주세요.",
      });
      return;
    }

    setBusy(true);
    try {
      const parsed = await parseResumeFile(file);
      const warningText =
        parsed.warnings.length > 0 ? `\n참고: ${parsed.warnings.join(" ")}` : "";
      push({
        role: "assistant",
        text: `${parsed.filename}에서 경력 텍스트 ${parsed.char_count.toLocaleString()}자를 추출했어요.${warningText}`,
      });
      await startProfile({ raw_text: parsed.text, current_job_title: null });
    } catch (e) {
      push({ role: "assistant", text: errorText(e), error: true });
    } finally {
      setBusy(false);
    }
  };

  // ── 설문 경로: 6문항 → 경력 서사 조립 → STEP 2 안내 ──────
  const handleSurveySubmit = async (survey: SurveyInput) => {
    push({
      role: "user",
      text: `설문으로 시작할게요. (${survey.job_title} · ${survey.years})`,
    });
    setBusy(true);
    try {
      // 백엔드가 설문을 경력 서사로 조립해 돌려주므로, 이후 단계는 이력서 경로와 동일해진다
      const { career_text } = await composeSurvey(survey);
      push({
        role: "assistant",
        text: "답변해주신 내용을 경력 서사로 정리했어요.",
      });
      await startProfile({ raw_text: career_text, current_job_title: survey.job_title });
    } catch (e) {
      push({ role: "assistant", text: errorText(e), error: true });
    } finally {
      setBusy(false);
    }
  };

  // ── STEP 2: 역량 분해 요청 ───────────────────────────────
  const handleAction = async (label: string) => {
    if (!career || draftSkills) return;
    push({ role: "user", text: label });
    setBusy(true);
    try {
      const profile = await generateProfile(career);
      setDraftSkills(profile.skills);
      // 이력서·자유 텍스트 경로는 직무명 입력이 없으므로, 서버가 경력 서사에서 뽑은 현재 직무를 기억해
      // STEP 3에서 '같은 직무' 후보를 걸러내는 데 쓴다
      if (!career.current_job_title && profile.current_job_title) {
        setCareer({ ...career, current_job_title: profile.current_job_title });
      }
      push(
        {
          role: "assistant",
          text: "경력 서사에서 다른 직무로도 옮겨갈 수 있는 역량들을 찾아냈어요.\n제가 잘못 읽었거나 빠뜨린 게 있다면 직접 고쳐주세요 — 최종 판단은 언제나 당신이 합니다.",
        },
        { role: "assistant", card: "skills" },
      );
    } catch (e) {
      push({ role: "assistant", text: errorText(e), error: true });
    } finally {
      setBusy(false);
    }
  };

  // ── STEP 2 확정 → STEP 3: 직무 매칭 ─────────────────────
  const handleSkillsConfirm = (confirmed: SkillItem[]) => {
    setConfirmedSkills(confirmed);
    push({ role: "user", text: `${confirmed.length}개 역량으로 프로필을 확정할게요.` });
    push(
      {
        role: "assistant",
        text: "좋아요. 이번에는 다음 직무를 찾는 방향을 먼저 정해볼게요.\n현재 경험을 활용해 인접 직무로 옮길지, 교육을 거쳐 완전히 다른 직무로 전환할지 선택해주세요.",
      },
      { role: "assistant", card: "job-path" },
    );
  };

  const handleJobPathSelect = async (track: JobSearchTrack) => {
    if (!confirmedSkills || track === jobSearchTrack) return;
    const trackLabel =
      track === "adjacent_transition" ? "인접 직무 전환형" : "교육 후 직무 전환형";
    setJobSearchTrack(track);
    setJobs(null);
    if (selectedJob) {
      setSelectedJob(null);
      setRoadmap(null);
      setDoneItems([]);
      setReflected(false);
      setReflectShown(false);
    }
    push({
      role: "user",
      text:
        jobSearchTrack === null
          ? `“${trackLabel}”으로 공고를 찾아볼게요.`
          : `“${trackLabel}”으로 다시 찾아볼게요.`,
    });
    setBusy(true);
    try {
      const result = await matchJobs(
        { skills: confirmedSkills },
        track,
        career?.current_job_title ?? undefined,
      );
      if (result.length > 0) {
        setJobs(result);
      } else {
        setJobs(null);
        setJobSearchTrack(null);
      }
      push(
        {
          role: "assistant",
          text:
            result.length > 0
              ? `${trackLabel} 기준으로 실제 채용공고를 선별했어요.\n적합도와 함께 수요 전망·전환 난이도·새로 배워야 할 역량을 살펴보세요.`
              : `현재 수집된 공고에서는 “${trackLabel}”에 맞는 후보를 찾지 못했어요. 다른 탐색 방향으로 다시 시도해주세요.`,
        },
        ...(result.length > 0 ? [{ role: "assistant" as const, card: "jobs" as const }] : []),
      );
    } catch (e) {
      setJobSearchTrack(null);
      push({ role: "assistant", text: errorText(e), error: true });
    } finally {
      setBusy(false);
    }
  };

  // ── STEP 3 선택 → STEP 4: 로드맵 ────────────────────────
  const handleJobSelect = async (job: JobMatch) => {
    if (!confirmedSkills) return;
    setSelectedJob(job);
    push({ role: "user", text: `"${job.job_title}"(으)로 가는 길을 보여주세요.` });
    setBusy(true);
    try {
      const result = await generateRoadmap(
        { skills: confirmedSkills },
        job.job_title,
        job.missing_skills,
      );
      setRoadmap(result);
      push(
        {
          role: "assistant",
          text: `${job.job_title}에 필요한 역량과 지금 가진 역량의 격차를 분석해 학습 로드맵을 그렸어요.\n각 항목의 근거에는 실제 NCS 능력단위·훈련과정이 표기됩니다.`,
        },
        { role: "assistant", card: "roadmap" },
        {
          role: "assistant",
          text: "이제 실행할 차례예요. 항목을 하나씩 완료 표시하면 여정 진행률에 바로 반영됩니다.\n속도와 경로는 당신이 정하면 돼요. 응원할게요! 🧭",
        },
      );
    } catch (e) {
      setSelectedJob(null);
      push({ role: "assistant", text: errorText(e), error: true });
    } finally {
      setBusy(false);
    }
  };

  // ── STEP 4 실행: 학습 항목 완료 토글 ────────────────────
  const toggleItem = (name: string) =>
    setDoneItems((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    );

  // ── 여정 지도 CTA ───────────────────────────────────────
  const handleJourneyAction = (target: ActionTarget) => {
    if (target === "composer") {
      dockRef.current?.querySelector("textarea")?.focus();
      return;
    }
    // 역량 분해를 아직 시작하지 않았다면 CTA가 곧 트리거가 된다
    if (target === "skills" && career && !draftSkills && !busy) {
      void handleAction("네, 제 역량을 분해해주세요");
      return;
    }
    const selector =
      target === "jobs" && jobs === null
        ? '[data-card="job-path"]'
        : `[data-card="${target}"]`;
    const el = scrollRef.current?.querySelector(selector);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const renderCard = (kind: CardKind) => {
    switch (kind) {
      case "risk":
        return risk ? <RiskDiagnosisCard data={risk} /> : null;
      case "skills":
        return draftSkills ? (
          <SkillProfileCard
            initial={draftSkills}
            locked={confirmedSkills !== null}
            onConfirm={handleSkillsConfirm}
          />
        ) : null;
      case "job-path":
        return (
          <JobPathChooser
            selected={jobSearchTrack}
            disabled={busy}
            onSelect={handleJobPathSelect}
          />
        );
      case "jobs":
        return jobs && jobSearchTrack ? (
          <JobMatchList
            jobs={jobs}
            searchTrack={jobSearchTrack}
            selected={selectedJob?.job_title ?? null}
            onSelect={handleJobSelect}
          />
        ) : null;
      case "roadmap":
        return roadmap ? (
          <RoadmapCard data={roadmap} done={doneItems} onToggle={toggleItem} />
        ) : null;
      case "reflect":
        return roadmap && selectedJob ? (
          <ReflectionCard
            fromJob={career?.current_job_title ?? "지금까지의 경력"}
            toJob={selectedJob.job_title}
            skillCount={confirmedSkills?.length ?? 0}
            itemCount={roadmap.items.length}
            totalWeeks={roadmap.items.reduce((s, it) => s + it.duration_weeks, 0)}
            completed={reflected}
            onComplete={() => setReflected(true)}
            onNewJourney={onNewChat}
          />
        ) : null;
    }
  };

  // 시작 화면의 파일 선택은 자유 입력 Composer와 독립적으로 동작한다.
  const openFilePicker = () => startFileRef.current?.click();

  // ── 시작 화면 · 설문 작성 ────────────────────────────────
  if (messages.length === 0 && startMode === "survey") {
    return (
      <main className="chat">
        <div className="survey-view">
          <div className="survey-inner">
            <SurveyForm
              onSubmit={handleSurveySubmit}
              onCancel={() => setStartMode("choose")}
              busy={busy}
            />
          </div>
        </div>
      </main>
    );
  }

  // ── 시작 화면 (히어로) ───────────────────────────────────
  if (messages.length === 0) {
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
              ref={startFileRef}
              className="file-input"
              type="file"
              accept=".pdf,.docx,.hwp,.hwpx"
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) handleFile(file);
                event.target.value = "";
              }}
            />

            <StartChooser
              onSurvey={() => setStartMode("survey")}
              onFile={openFilePicker}
              disabled={busy}
            />

            {/* 시작 전에도 앞으로의 여정을 미리 보여준다 */}
            <JourneyMap journey={journey} onAction={handleJourneyAction} preview />

            <p className="hero-disclaimer">
              AI가 판정하지 않습니다 — 모든 단계는 당신이 검토하고 선택합니다.
            </p>
          </div>
        </div>
      </main>
    );
  }

  // ── 대화 화면 ────────────────────────────────────────────
  return (
    <main className="chat">
      {/* 결과를 읽는 흐름을 방해하지 않도록 여정은 요약으로 시작하고 필요할 때 펼친다. */}
      <div className={`journey-dock${journeyExpanded ? " expanded" : ""}`}>
        <button
          type="button"
          className="journey-toggle"
          onClick={() => setJourneyExpanded((expanded) => !expanded)}
          aria-expanded={journeyExpanded}
          aria-controls="journey-detail"
        >
          <span className="journey-toggle-context">
            <span className="journey-toggle-eyebrow">나의 여정</span>
            <span className="journey-toggle-stage">
              {journey.complete ? "여정 완주" : `${journey.current.step}단계 · ${journey.current.label}`}
            </span>
          </span>
          <span className="journey-toggle-progress" aria-label={`전체 달성률 ${journey.percent}%`}>
            <span className="journey-toggle-bar" aria-hidden>
              <span style={{ width: `${journey.percent}%` }} />
            </span>
            <b>{journey.percent}%</b>
          </span>
          <span className="journey-toggle-action">
            {journeyExpanded ? "접기" : "전체 여정 보기"}
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
              <path
                d="m4 6 4 4 4-4"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
        </button>
        {journeyExpanded && (
          <div id="journey-detail">
            <JourneyMap journey={journey} onAction={handleJourneyAction} />
          </div>
        )}
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-col">
          {messages.map((m) => (
            <div className={`msg ${m.role}`} key={m.id} data-card={m.card}>
              {m.role === "assistant" && (
                <div className="msg-avatar">
                  <CompassMark size={18} />
                </div>
              )}
              <div className={`msg-body${m.error ? " msg-error" : ""}`}>
                {m.text}
                {m.card && renderCard(m.card)}
                {m.action && (
                  <div className="actions">
                    <button
                      className="action-chip"
                      disabled={draftSkills !== null || busy}
                      onClick={() => handleAction(m.action!)}
                    >
                      {m.action}
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          {busy && (
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
                {busySeconds >= 10 && (
                  <p className="busy-note">
                    분석 중이에요 · {Math.floor(busySeconds / 60)}분 {busySeconds % 60}초 경과
                    <br />
                    로컬 AI 모델은 단계에 따라 최대 20분 정도 걸릴 수 있어요. 창을 닫지 말고 기다려주세요.
                  </p>
                )}
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>
      <div className="composer-dock" ref={dockRef}>
        <Composer
          onSend={handleSend}
          onFileSelect={handleFile}
          disabled={busy}
          placeholder="메시지를 입력하거나 이력서를 첨부하세요…"
        />
        <p className="composer-hint">
          PDF · DOCX · HWP · HWPX (최대 10MB) 또는 자유 텍스트를 지원합니다
        </p>
      </div>
    </main>
  );
}
