import { ExternalLink, FileText, X } from "lucide-react";
import { useEffect } from "react";

import type { Source } from "../../api/types";

interface SourceDrawerProps {
  sources: Source[];
  open: boolean;
  onClose(): void;
}

export function SourceDrawer({ sources, open, onClose }: SourceDrawerProps) {
  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, open]);

  if (!open) return null;
  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside
        aria-label="Sumber hukum"
        aria-modal="true"
        className="source-drawer"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
      >
        <header className="drawer-header">
          <div>
            <span className="eyebrow">Dasar jawaban</span>
            <h2>Sumber hukum</h2>
          </div>
          <button className="icon-button" onClick={onClose} type="button" aria-label="Tutup sumber">
            <X aria-hidden="true" size={20} />
          </button>
        </header>
        <div className="source-list">
          {sources.map((source) => (
            <article className="source-item" id={`source-${source.source_id}`} key={source.source_id}>
              <div className="source-title-row">
                <span className="source-id">{source.source_id}</span>
                <div>
                  <h3>
                    Pasal {source.article_number}
                    {source.paragraph_number ? ` ayat (${source.paragraph_number})` : ""}
                  </h3>
                  <p>{source.chunk_type === "elucidation" ? "Penjelasan" : "Batang tubuh"}</p>
                </div>
              </div>
              <blockquote>{source.excerpt}</blockquote>
              <div className="source-meta">
                <span><FileText size={14} /> {source.document_title}</span>
                <a href={source.official_url} target="_blank" rel="noreferrer">
                  Buka dokumen resmi <ExternalLink size={13} />
                </a>
              </div>
              <small>Diverifikasi {source.last_verified_at}</small>
            </article>
          ))}
        </div>
      </aside>
    </div>
  );
}
