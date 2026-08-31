import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useChat } from "./useChat";

const SESSION_KEY = "tanya-lalin.session-id";

function historyResponse(): Response {
  return new Response(
    JSON.stringify({
      session_id: "saved-session",
      messages: [
        {
          id: "assistant-1",
          role: "assistant",
          content: "Jawaban tersimpan [S1].",
          status: "answered",
          citations: [{ citation_id: "C1", marker: "[S1]", source_id: "S1" }],
          sources: [],
          created_at: "2026-08-27T00:00:00Z",
        },
      ],
      created_at: "2026-08-27T00:00:00Z",
      updated_at: "2026-08-27T00:00:00Z",
    }),
    { status: 200 },
  );
}

describe("useChat", () => {
  beforeEach(() => localStorage.clear());

  it("restores history and clears both local and server sessions", async () => {
    localStorage.setItem(SESSION_KEY, "saved-session");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(historyResponse())
      .mockResolvedValueOnce(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => useChat());

    await waitFor(() => expect(result.current.restoring).toBe(false));
    expect(result.current.messages[0].content).toContain("tersimpan");
    await act(() => result.current.newChat());
    expect(result.current.messages).toEqual([]);
    expect(localStorage.getItem(SESSION_KEY)).toBeNull();
    expect(fetchMock).toHaveBeenLastCalledWith("/api/v2/chat/saved-session", {
      method: "DELETE",
    });
  });

  it("clears an expired saved session", async () => {
    localStorage.setItem(SESSION_KEY, "expired");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({ detail: "expired", error_code: "session_not_found" }),
          { status: 404 },
        ),
      ),
    );
    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.restoring).toBe(false));
    expect(result.current.notice).toContain("telah berakhir");
    expect(localStorage.getItem(SESSION_KEY)).toBeNull();
  });

  it("prevents concurrent submissions and supports cancellation", async () => {
    const fetchMock = vi.fn((_url: string, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.restoring).toBe(false));

    act(() => {
      void result.current.send("first");
      void result.current.send("second");
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    act(() => result.current.cancel());
    await waitFor(() => expect(result.current.pending).toBe(false));
    expect(result.current.error).toBeNull();
  });

  it.each([
    [429, "rate_limited", "Terlalu banyak permintaan"],
    [503, "provider_unavailable", "Layanan jawaban sedang tidak tersedia"],
  ])("shows a specific error for HTTP %s", async (status, code, message) => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "error", error_code: code }), { status }),
      ),
    );
    const { result } = renderHook(() => useChat());
    await waitFor(() => expect(result.current.restoring).toBe(false));
    await act(() => result.current.send("question"));
    expect(result.current.error).toContain(message);
  });
});
