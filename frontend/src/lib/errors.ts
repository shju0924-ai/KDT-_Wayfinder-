// API 오류 → 사용자에게 보여줄 메시지.

/** 사용자에게 보여줄 오류 메시지로 변환 (원인을 숨기지 않되 읽을 수 있게).
 *  4xx 는 입력 문제라 서버 상태 안내를 붙이지 않는다. 파일 업로드에서 난 4xx 만 '파일' 문구를 쓴다. */
export function errorText(e: unknown, source: "file" | "request" = "request"): string {
  const status = (e as { response?: { status?: number } })?.response?.status;
  const rawDetail = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  // FastAPI 요청 검증 오류(422)는 detail 이 객체 배열이라 그대로 넣으면 [object Object] 가 된다
  const detail =
    typeof rawDetail === "string"
      ? rawDetail
      : rawDetail !== undefined
        ? JSON.stringify(rawDetail)
        : ((e as Error)?.message ?? String(e));
  if (status && status < 500) {
    return source === "file"
      ? `파일을 읽지 못했어요.\n(${detail})`
      : `요청 내용을 처리하지 못했어요.\n(${detail})`;
  }
  return `요청을 처리하지 못했어요.\n(${detail})\n\n백엔드가 실행 중인지, API 키가 설정돼 있는지 확인해주세요.`;
}
