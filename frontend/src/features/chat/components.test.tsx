import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { UiMessage } from "./useChat";
import { AnswerCard } from "./AnswerCard";
import { Composer } from "./Composer";
import { ProgressIndicator } from "./ProgressIndicator";

const assistant: UiMessage = {
  id: "message",
  role: "assistant",
  content: "Pelanggar dapat dikenai denda [S1].",
  createdAt: "2026-08-27T00:00:00Z",
  status: "answered",
  citations: [{ citation_id: "C1", marker: "[S1]", source_id: "S1" }],
  sources: [
    {
      source_id: "S1",
      document_id: "law",
      document_title: "Law No. 22 of 2009",
      article_number: 287,
      paragraph_number: 2,
      chunk_type: "body",
      excerpt: "Denda paling banyak Rp500.000.",
      official_url: "https://example.test/law",
      last_verified_at: "2026-08-27",
    },
  ],
};

describe("chat components", () => {
  it("renders citation chips and an accessible source drawer", async () => {
    const user = userEvent.setup();
    render(<AnswerCard message={assistant} />);
    await user.click(screen.getByRole("button", { name: "[S1]" }));
    expect(screen.getByRole("dialog", { name: "Sumber hukum" })).toBeInTheDocument();
    expect(screen.getByText("Pasal 287 ayat (2)")).toBeInTheDocument();
    expect(screen.getByText("Denda paling banyak Rp500.000.")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Tutup sumber" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it.each([
    ["blocked", "Pertanyaan dibatasi"],
    ["insufficient_evidence", "Bukti belum cukup"],
  ] as const)("renders the %s response state", (status, label) => {
    render(<AnswerCard message={{ ...assistant, status, citations: [], sources: [] }} />);
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("supports Enter, Shift+Enter, pending cancellation, and accessible names", async () => {
    const onSend = vi.fn();
    const onCancel = vi.fn();
    const { rerender } = render(<Composer pending={false} onSend={onSend} onCancel={onCancel} />);
    const input = screen.getByRole("textbox", { name: "Tulis pertanyaan lalu lintas" });
    fireEvent.change(input, { target: { value: "Pertanyaan" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSend).toHaveBeenCalledWith("Pertanyaan");

    rerender(<Composer pending onSend={onSend} onCancel={onCancel} />);
    await userEvent.click(screen.getByRole("button", { name: "Batalkan jawaban" }));
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it("announces the current pipeline stage", () => {
    render(<ProgressIndicator stage="retrieving" />);
    expect(screen.getByRole("status")).toHaveTextContent("Mencari dasar hukum");
  });
});
