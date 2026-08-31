import { BookOpenCheck, Plus, ShieldCheck, Sparkles } from "lucide-react";
import { useLayoutEffect, useRef } from "react";

import { AnswerCard } from "./features/chat/AnswerCard";
import { Composer } from "./features/chat/Composer";
import { ProgressIndicator } from "./features/chat/ProgressIndicator";
import { useChat } from "./features/chat/useChat";

const SUGGESTIONS = [
  "Apa sanksi menerobos lampu merah?",
  "Apakah pengendara motor wajib memakai helm SNI?",
  "Apa aturan menggunakan ponsel saat mengemudi?",
];

export default function App() {
  const chat = useChat();
  const endRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (chat.messages.length === 0) return;

    const frame = window.requestAnimationFrame(() => {
      window.scrollTo({
        top: document.documentElement.scrollHeight,
        behavior: chat.stage ? "auto" : "smooth",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [chat.messages, chat.stage]);

  return (
    <div className={`app-shell ${chat.messages.length === 0 ? "is-empty" : ""}`}>
      <header className="topbar">
        <a className="brand" href="/" aria-label="Tanya Lalin, beranda">
          <span className="brand-mark"><BookOpenCheck aria-hidden="true" size={22} /></span>
          <span><strong>Tanya Lalin</strong><small>Asisten aturan lalu lintas</small></span>
        </a>
        <button className="new-chat" onClick={() => void chat.newChat()} type="button">
          <Plus aria-hidden="true" size={17} /> Percakapan baru
        </button>
      </header>

      <main className="chat-layout">
        <section className="chat-column" aria-label="Percakapan">
          {chat.restoring ? (
            <div className="restore-state" role="status">Memulihkan percakapan…</div>
          ) : chat.messages.length === 0 ? (
            <div className="welcome">
              <span className="welcome-icon"><Sparkles aria-hidden="true" size={25} /></span>
              <p className="eyebrow">Jawaban berbasis sumber resmi</p>
              <h1>Pahami aturan jalan,<br />tanpa menebak-nebak.</h1>
              <p className="welcome-copy">
                Tanyakan kewajiban, larangan, atau sanksi dalam UU Lalu Lintas dan Angkutan Jalan.
                Setiap jawaban yang diberikan akan melewati pemeriksaan sumber.
              </p>
              <div className="trust-row">
                <span><ShieldCheck size={15} /> Kutipan terverifikasi</span>
                <span><BookOpenCheck size={15} /> Sumber resmi</span>
              </div>
              <div className="suggestions" aria-label="Contoh pertanyaan">
                {SUGGESTIONS.map((suggestion) => (
                  <button key={suggestion} onClick={() => void chat.send(suggestion)} type="button">
                    {suggestion}<span aria-hidden="true">→</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages" aria-live="polite">
              {chat.messages.map((message) =>
                message.role === "user" ? (
                  <div className="user-row" key={message.id}>
                    <div className="user-message">{message.content}</div>
                  </div>
                ) : (
                  <AnswerCard key={message.id} message={message} />
                ),
              )}
              {chat.stage && <ProgressIndicator stage={chat.stage} />}
              <div className="chat-end" ref={endRef} aria-hidden="true" />
            </div>
          )}
          {(chat.notice || chat.error) && (
            <div className={chat.error ? "inline-alert error" : "inline-alert"} role={chat.error ? "alert" : "status"}>
              {chat.error ?? chat.notice}
            </div>
          )}
        </section>
      </main>

      <footer className="composer-wrap">
        <div className="composer-inner">
          <Composer pending={chat.pending} onSend={(message) => void chat.send(message)} onCancel={chat.cancel} />
          <p className="scope-note">
            Referensi saat ini: UU No. 22 Tahun 2009 versi asli dan penjelasannya, belum mengonsolidasikan perubahan.
          </p>
        </div>
      </footer>
    </div>
  );
}
