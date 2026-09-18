// 메시지 입력창 — Enter 전송, Shift+Enter 줄바꿈, 한글 IME 조합 중 전송 방지
import { useRef, useState } from "react";
import { AttachmentIcon, SendIcon } from "./icons";

interface Props {
  onSend: (text: string) => void;
  onFileSelect?: (file: File) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function Composer({ onSend, onFileSelect, disabled, placeholder }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText("");
    if (ref.current) ref.current.style.height = "auto";
  };

  return (
    <div className="composer">
      {onFileSelect && (
        <>
          <input
            ref={fileRef}
            className="file-input"
            type="file"
            accept=".pdf,.docx,.hwp,.hwpx"
            disabled={disabled}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onFileSelect(file);
              event.target.value = "";
            }}
          />
          <button
            type="button"
            className="attach-btn"
            disabled={disabled}
            onClick={() => fileRef.current?.click()}
            aria-label="이력서 파일 첨부"
            title="이력서 첨부 (PDF, DOCX, HWP, HWPX · 최대 10MB)"
          >
            <AttachmentIcon />
          </button>
        </>
      )}
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder={placeholder ?? "지금까지 해온 일을 자유롭게 들려주세요…"}
        onChange={(e) => {
          setText(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${e.target.scrollHeight}px`;
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <button
        className="send-btn"
        onClick={submit}
        disabled={disabled || !text.trim()}
        aria-label="보내기"
      >
        <SendIcon />
      </button>
    </div>
  );
}
