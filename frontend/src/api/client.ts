import { createParser } from "eventsource-parser";

import type {
  ChatResponse,
  PipelineStage,
  ProblemDetails,
  SessionHistory,
} from "./types";

const STAGES = new Set<PipelineStage>([
  "validating",
  "analyzing",
  "retrieving",
  "reranking",
  "generating",
  "verifying",
]);

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function readProblem(response: Response): Promise<ApiError> {
  try {
    const problem = (await response.json()) as ProblemDetails;
    return new ApiError(problem.detail, response.status, problem.error_code, problem.request_id);
  } catch {
    return new ApiError("Server returned an invalid error response.", response.status, "invalid_response");
  }
}

export interface StreamHandlers {
  onSession(sessionId: string): void;
  onStatus(stage: PipelineStage): void;
}

export async function streamChat(
  message: string,
  sessionId: string | null,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });
  if (!response.ok) throw await readProblem(response);
  if (!response.body) throw new ApiError("Streaming is unavailable.", 0, "connection_interrupted");

  let complete: ChatResponse | null = null;
  let streamError: ApiError | null = null;
  const parser = createParser({
    onEvent: ({ event, data }) => {
      try {
        const payload: unknown = JSON.parse(data);
        if (event === "meta") {
          const meta = payload as { session_id?: unknown };
          if (typeof meta.session_id !== "string") throw new Error("Invalid meta event");
          handlers.onSession(meta.session_id);
        } else if (event === "status") {
          const status = payload as { stage?: unknown };
          if (typeof status.stage !== "string" || !STAGES.has(status.stage as PipelineStage)) {
            throw new Error("Invalid status event");
          }
          handlers.onStatus(status.stage as PipelineStage);
        } else if (event === "complete") {
          complete = payload as ChatResponse;
        } else if (event === "error") {
          const problem = payload as ProblemDetails;
          streamError = new ApiError(
            problem.detail,
            problem.status,
            problem.error_code,
            problem.request_id,
          );
        }
      } catch (error) {
        streamError = new ApiError(
          error instanceof Error ? error.message : "Malformed stream event.",
          0,
          "malformed_event",
        );
      }
    },
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.feed(decoder.decode(value, { stream: true }));
    if (streamError) throw streamError;
  }
  parser.feed(decoder.decode());
  if (streamError) throw streamError;
  if (!complete) {
    throw new ApiError("The connection closed before an answer arrived.", 0, "connection_interrupted");
  }
  return complete;
}

export async function getHistory(sessionId: string, signal: AbortSignal): Promise<SessionHistory> {
  const response = await fetch(`/api/v1/chat/${encodeURIComponent(sessionId)}/history`, { signal });
  if (!response.ok) throw await readProblem(response);
  return (await response.json()) as SessionHistory;
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/v1/chat/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok && response.status !== 404) throw await readProblem(response);
}
