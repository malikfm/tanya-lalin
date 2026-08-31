export type AnswerStatus = "answered" | "blocked" | "insufficient_evidence";

export type PipelineStage =
  | "validating"
  | "analyzing"
  | "retrieving"
  | "reranking"
  | "generating"
  | "verifying";

export interface Citation {
  citation_id: string;
  marker: string;
  source_id: string;
}

export interface Source {
  source_id: string;
  document_id: string;
  document_title: string;
  article_number: number;
  paragraph_number: number | null;
  chunk_type: "body" | "elucidation";
  excerpt: string;
  official_url: string;
  last_verified_at: string;
}

export interface ChatResponse {
  request_id: string;
  session_id: string;
  message_id: string;
  status: AnswerStatus;
  answer: string;
  citations: Citation[];
  sources: Source[];
  created_at: string;
}

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: AnswerStatus | null;
  citations: Citation[];
  sources: Source[];
  created_at: string;
}

export interface SessionHistory {
  session_id: string;
  messages: StoredMessage[];
  created_at: string;
  updated_at: string;
}

export interface ProblemDetails {
  type: string;
  title: string;
  status: number;
  detail: string;
  error_code: string;
  request_id: string;
}
