import { BookOpen, CircleAlert, ShieldCheck } from "lucide-react";
import { useCallback, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";

import type { Citation } from "../../api/types";
import type { UiMessage } from "./useChat";
import { SourceDrawer } from "./SourceDrawer";

function withCitationLinks(answer: string, citations: Citation[]): string {
  const known = new Set(citations.map((citation) => citation.marker));
  return answer.replace(/\[S\d+\]/g, (marker) =>
    known.has(marker) ? `[${marker}](citation:${marker.slice(1, -1)})` : marker,
  );
}

export function AnswerCard({ message }: { message: UiMessage }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const citations = message.citations ?? [];
  const sources = message.sources ?? [];
  const openSource = useCallback(
    (sourceId: string) => {
      setDrawerOpen(true);
      window.setTimeout(
        () => document.getElementById(`source-${sourceId}`)?.scrollIntoView?.(),
        0,
      );
    },
    [],
  );
  const isBlocked = message.status === "blocked";
  const isInsufficient = message.status === "insufficient_evidence";

  return (
    <article className={`answer-card ${isBlocked || isInsufficient ? "answer-warning" : ""}`}>
      <div className="answer-label">
        {isBlocked || isInsufficient ? <CircleAlert size={16} /> : <ShieldCheck size={16} />}
        <span>
          {isBlocked
            ? "Pertanyaan dibatasi"
            : isInsufficient
              ? "Bukti belum cukup"
              : "Jawaban terverifikasi"}
        </span>
      </div>
      <div className="answer-content">
        <ReactMarkdown
          components={{
            a: ({ href, children }) => {
              if (href?.startsWith("citation:")) {
                const sourceId = href.slice("citation:".length);
                return (
                  <button className="citation-chip" onClick={() => openSource(sourceId)} type="button">
                    {children}
                  </button>
                );
              }
              return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
            },
          }}
          urlTransform={(url) =>
            url.startsWith("citation:") ? url : defaultUrlTransform(url)
          }
        >
          {withCitationLinks(message.content, citations)}
        </ReactMarkdown>
      </div>
      {sources.length > 0 && (
        <button className="sources-button" onClick={() => setDrawerOpen(true)} type="button">
          <BookOpen aria-hidden="true" size={16} />
          Lihat {sources.length} sumber hukum
        </button>
      )}
      <SourceDrawer sources={sources} open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </article>
  );
}
