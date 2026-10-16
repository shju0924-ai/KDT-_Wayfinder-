// STEP 2 — 역량 프로필 카드. 핵심 UX: AI가 경력에서 찾은 NCS 능력단위를 사용자가 직접 검토·수정·확정한다.
// 능력단위마다 근거가 된 경력 구절을 보여 주고, 빠진 것은 추천 후보에서 고르거나 NCS 검색으로 찾아 추가한다.
import { useState } from "react";
import { searchNcsUnits } from "../../api/client";
import type { SkillItem } from "../../types/api";
import { CheckIcon, PlusIcon, XIcon } from "../icons";

interface Props {
  initial: SkillItem[];
  /** 검색 후보 중 AI가 고르지 않은 능력단위 — evidence 는 후보를 찾은 경력 구절 */
  suggestions?: SkillItem[];
  locked: boolean;
  onConfirm: (skills: SkillItem[]) => void;
}

export default function SkillProfileCard({ initial, suggestions = [], locked, onConfirm }: Props) {
  const [skills, setSkills] = useState<SkillItem[]>(initial);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SkillItem[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  const held = new Set(skills.map((s) => s.ncs_code));
  const remove = (code: string) => setSkills((s) => s.filter((x) => x.ncs_code !== code));
  // 검색으로 찾은 단위는 경력 근거가 없고, 추천 후보는 후보를 찾은 경력 구절을 근거로 그대로 가져간다
  const add = (unit: SkillItem, keepEvidence = false) => {
    if (held.has(unit.ncs_code)) return;
    setSkills((s) => [
      ...s,
      { ...unit, user_added: true, evidence: keepEvidence ? unit.evidence : null, confirmed: false },
    ]);
  };
  const openSuggestions = suggestions.filter((u) => !held.has(u.ncs_code));

  const search = async () => {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    setSearchError(null);
    try {
      setResults(await searchNcsUnits(q));
    } catch {
      setSearchError("검색하지 못했어요. 잠시 후 다시 시도해 주세요.");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="card">
      <div className="card-eyebrow">STEP 2 · 역량 프로필</div>
      <h3>경력에서 확인한 직무 능력</h3>
      <p className="skill-intro">
        국가직무능력표준(NCS)의 능력단위로 정리했어요. 경력의 어느 구절에서 찾았는지 함께 보여 드려요.
      </p>
      <div className="skill-list">
        {skills.map((s) => (
          <div className="skill-row" key={s.ncs_code}>
            <div className="skill-main">
              <div className="skill-name">
                {s.name}
                <span className="tag" title={s.ncs_path}>
                  {s.ncs_job}
                </span>
              </div>
              {!s.evidence ? (
                <div className="skill-evidence">직접 추가 — {s.definition}</div>
              ) : s.user_added ? (
                <div className="skill-evidence">직접 선택 · 근거 — “{s.evidence}”</div>
              ) : (
                <div className="skill-evidence">근거 — “{s.evidence}”</div>
              )}
            </div>
            {!locked && (
              <button
                className="skill-remove"
                onClick={() => remove(s.ncs_code)}
                aria-label={`${s.name} 삭제`}
              >
                <XIcon />
              </button>
            )}
          </div>
        ))}
        {skills.length === 0 && (
          <div className="skill-evidence">
            경력에서 찾은 능력단위가 없어요. 아래에서 해 본 일을 검색해 추가해 주세요.
          </div>
        )}
      </div>

      {!locked && openSuggestions.length > 0 && (
        <div className="skill-suggest">
          <div className="skill-suggest-head">이 능력도 해당되나요?</div>
          <p className="skill-evidence">
            경력 구절로 찾았지만 AI가 확신하지 못해 고르지 않은 후보예요. 해 본 일이 맞다면 추가하세요.
          </p>
          <div className="skill-results">
            {openSuggestions.map((u) => (
              <div className="skill-row" key={u.ncs_code}>
                <div className="skill-main">
                  <div className="skill-name">
                    {u.name}
                    <span className="tag" title={u.ncs_path}>
                      {u.ncs_job}
                    </span>
                  </div>
                  <div className="skill-evidence">{u.definition}</div>
                  {u.evidence && <div className="skill-evidence">찾은 구절 — “{u.evidence}”</div>}
                </div>
                <button className="btn ghost" onClick={() => add(u, true)} aria-label={`${u.name} 추가`}>
                  <PlusIcon />
                  추가
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {!locked && (
        <>
          <div className="skill-add">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && search()}
              placeholder="빠진 능력이 있다면 검색해 추가하세요 (예: 채권관리, 재고관리)"
            />
            <button className="btn ghost" onClick={search} disabled={!query.trim() || searching}>
              {searching ? "검색 중…" : "검색"}
            </button>
          </div>
          {searchError && <div className="skill-evidence">{searchError}</div>}
          {results && (
            <div className="skill-results">
              {results.length === 0 && <div className="skill-evidence">찾은 능력단위가 없어요.</div>}
              {results.map((u) => (
                <div className="skill-row" key={u.ncs_code}>
                  <div className="skill-main">
                    <div className="skill-name">
                      {u.name}
                      <span className="tag" title={u.ncs_path}>
                        {u.ncs_job}
                      </span>
                    </div>
                    <div className="skill-evidence">{u.definition}</div>
                  </div>
                  <button
                    className="btn ghost"
                    onClick={() => add(u)}
                    disabled={held.has(u.ncs_code)}
                    aria-label={`${u.name} 추가`}
                  >
                    {held.has(u.ncs_code) ? <CheckIcon /> : <PlusIcon />}
                    {held.has(u.ncs_code) ? "추가됨" : "추가"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <div className="card-foot">
        <span className="note">
          {locked
            ? `${skills.length}개 능력으로 확정됨`
            : "확정 전에 자유롭게 삭제·추가할 수 있어요. 선택은 언제나 당신의 몫이에요."}
        </span>
        {locked ? (
          <span className="confirmed-note">
            <CheckIcon /> 확정 완료
          </span>
        ) : (
          <button
            className="btn"
            disabled={skills.length === 0}
            onClick={() => onConfirm(skills.map((s) => ({ ...s, confirmed: true })))}
          >
            이 역량 프로필로 확정
          </button>
        )}
      </div>
    </div>
  );
}
