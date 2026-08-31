import { ArrowUp, Square } from "lucide-react";
import { FormEvent, KeyboardEvent, useState } from "react";

interface ComposerProps {
  pending: boolean;
  onSend(message: string): void;
  onCancel(): void;
}

export function Composer({ pending, onSend, onCancel }: ComposerProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const message = value.trim();
    if (!message || pending) return;
    setValue("");
    onSend(message);
  };
  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit();
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form className="composer" onSubmit={onSubmit}>
      <label className="sr-only" htmlFor="chat-message">Tulis pertanyaan lalu lintas</label>
      <textarea
        autoFocus
        id="chat-message"
        maxLength={2000}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Contoh: Apa sanksi menerobos lampu merah?"
        rows={2}
        value={value}
      />
      {pending ? (
        <button className="send-button cancel" onClick={onCancel} type="button" aria-label="Batalkan jawaban">
          <Square aria-hidden="true" size={16} fill="currentColor" />
        </button>
      ) : (
        <button className="send-button" disabled={!value.trim()} type="submit" aria-label="Kirim pertanyaan">
          <ArrowUp aria-hidden="true" size={20} />
        </button>
      )}
      <div className="composer-hint">
        <span>Enter untuk kirim · Shift+Enter untuk baris baru</span>
        <span>{value.length}/2000</span>
      </div>
    </form>
  );
}
