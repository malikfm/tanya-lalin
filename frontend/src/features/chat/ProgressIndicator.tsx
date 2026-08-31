import { Check, LoaderCircle } from "lucide-react";

import type { PipelineStage } from "../../api/types";

const FLOW: Array<{ stage: PipelineStage; label: string }> = [
  { stage: "analyzing", label: "Memahami pertanyaan" },
  { stage: "retrieving", label: "Mencari dasar hukum" },
  { stage: "generating", label: "Menyusun jawaban" },
  { stage: "verifying", label: "Memeriksa kutipan" },
];

function position(stage: PipelineStage): number {
  if (stage === "validating") return 0;
  if (stage === "reranking") return 1;
  return Math.max(
    FLOW.findIndex((item) => item.stage === stage),
    0,
  );
}

export function ProgressIndicator({ stage }: { stage: PipelineStage }) {
  const active = position(stage);
  return (
    <div className="progress-card" role="status" aria-live="polite">
      <div className="progress-heading">
        <LoaderCircle aria-hidden="true" size={17} />
        <span>Sedang menyiapkan jawaban terverifikasi</span>
      </div>
      <ol className="progress-steps" aria-label="Progres jawaban">
        {FLOW.map((item, index) => (
          <li className={index <= active ? "active" : ""} key={item.stage}>
            <span className="step-dot">{index < active ? <Check size={11} /> : index + 1}</span>
            {item.label}
          </li>
        ))}
      </ol>
    </div>
  );
}
