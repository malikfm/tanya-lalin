import { describe, expect, it, vi } from "vitest";

import { ApiError, getHistory, streamChat } from "./client";

function streamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );
}

const COMPLETE = {
  request_id: "request",
  session_id: "session",
  message_id: "message",
  status: "answered",
  answer: "Jawaban [S1].",
  citations: [],
  sources: [],
  created_at: "2026-08-27T00:00:00Z",
};

describe("API client", () => {
  it("reports statuses and returns only the atomic completion", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          'event: meta\ndata: {"session_id":"session"}\n\n',
          'event: status\ndata: {"stage":"analyzing"}\n\n',
          'event: status\ndata: {"stage":"verifying"}\n\n',
          `event: complete\ndata: ${JSON.stringify(COMPLETE)}\n\n`,
        ]),
      ),
    );
    const sessions: string[] = [];
    const statuses: string[] = [];
    const result = await streamChat(
      "test",
      null,
      { onSession: (id) => sessions.push(id), onStatus: (stage) => statuses.push(stage) },
      new AbortController().signal,
    );
    expect(sessions).toEqual(["session"]);
    expect(statuses).toEqual(["analyzing", "verifying"]);
    expect(result.answer).toBe("Jawaban [S1].");
  });

  it.each([
    ['event: status\ndata: {"stage":"unknown"}\n\n', "malformed_event"],
    ['event: meta\ndata: {}\n\n', "malformed_event"],
    ['event: status\ndata: {not-json}\n\n', "malformed_event"],
    ['event: status\ndata: {"stage":"analyzing"}\n\n', "connection_interrupted"],
  ])("rejects malformed or incomplete streams", async (body, code) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([body])));
    await expect(
      streamChat(
        "test",
        null,
        { onSession: vi.fn(), onStatus: vi.fn() },
        new AbortController().signal,
      ),
    ).rejects.toMatchObject({ code });
  });

  it.each([
    [429, "rate_limited"],
    [503, "provider_unavailable"],
  ])("preserves RFC 7807 errors", async (status, errorCode) => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ detail: "Unavailable", error_code: errorCode, request_id: "r" }),
          { status },
        ),
      ),
    );
    await expect(getHistory("session", new AbortController().signal)).rejects.toEqual(
      new ApiError("Unavailable", status, errorCode, "r"),
    );
  });
});
