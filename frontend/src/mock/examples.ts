// 시작 화면의 예시 입력 — 사용자가 무엇을 적어야 할지 감을 잡도록 돕는 UI 보조 텍스트.
// 결과가 아니라 '입력 예시'이므로 실제 API 응답과 무관하다.
// 원본은 examples.json — 백엔드 E2E 회귀(backend/scripts/e2e_eval.py --set ui)가 같은 파일을 읽어
// 화면에 내건 예시가 실제로 끝까지 잘 도는지 검증한다. 문구는 JSON 쪽에서 고칠 것.
import examples from "./examples.json";

export interface Suggestion {
  id: string;
  label: string;
  text: string;
}

export const suggestions: Suggestion[] = examples;
