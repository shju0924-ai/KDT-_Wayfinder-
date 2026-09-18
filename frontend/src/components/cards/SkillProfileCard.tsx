// STEP 2 — 역량 프로필 카드. 핵심 UX: AI가 분해한 역량을 사용자가 직접 검토·수정·확정한다.
import { useState } from "react";
import type { SkillItem } from "../../types/api";
import { CheckIcon, PlusIcon, XIcon } from "../icons";

interface Props {
  initial: SkillItem[];
  locked: boolean;
  onConfirm: (skills: SkillItem[]) => void;
}

export default function SkillProfileCard({ initial, locked, onConfirm }: Props) {
  const [skills, setSkills] = useState<SkillItem[]>(initial);
  const [draft, setDraft] = useState("");

  const remove = (name: string) => setSkills((s) => s.filter((x) => x.name !== name));

  const add = () => {
    const name = draft.trim();
    if (!name || skills.some((s) => s.name === name)) return;
    setSkills((s) => [
      ...s,
      { name, category: "직접 추가", evidence: "사용자가 직접 추가한 역량", confirmed: false },
    ]);
    setDraft("");
  };

  return (
    <div className="card">
      <div className="card-eyebrow">STEP 2 · 역량 프로필</div>
      <h3>경력에서 발견한 전이 가능 역량</h3>
      <div className="skill-list">
        {skills.map((s) => (
          <div className="skill-row" key={s.name}>
            <div className="skill-main">
              <div className="skill-name">
                {s.name}
                <span className="tag">{s.category}</span>
              </div>
              <div className="skill-evidence">근거 — {s.evidence}</div>
            </div>
            {!locked && (
              <button
                className="skill-remove"
                onClick={() => remove(s.name)}
                aria-label={`${s.name} 삭제`}
              >
                <XIcon />
              </button>
            )}
          </div>
        ))}
      </div>

      {!locked && (
        <div className="skill-add">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && add()}
            placeholder="빠진 역량이 있다면 직접 추가하세요"
          />
          <button className="btn ghost" onClick={add} disabled={!draft.trim()}>
            <PlusIcon /> 추가
          </button>
        </div>
      )}

      <div className="card-foot">
        <span className="note">
          {locked
            ? `${skills.length}개 역량으로 확정됨`
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
