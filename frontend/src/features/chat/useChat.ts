import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, deleteSession, getHistory, streamChat } from "../../api/client";
import type {
  AnswerStatus,
  ChatResponse,
  Citation,
  PipelineStage,
  Source,
} from "../../api/types";

const SESSION_KEY = "tanya-lalin.session-id";

export interface UiMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  status?: AnswerStatus;
  citations?: Citation[];
  sources?: Source[];
}

function errorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) {
    return "Koneksi terputus. Periksa jaringan Anda, lalu coba lagi.";
  }
  if (error.status === 429) return "Terlalu banyak permintaan. Tunggu sebentar sebelum mencoba lagi.";
  if (error.status === 503) return "Layanan jawaban sedang tidak tersedia. Silakan coba beberapa saat lagi.";
  if (error.code === "malformed_event") return "Respons server tidak dapat dibaca dengan aman.";
  return "Koneksi terputus sebelum jawaban selesai. Silakan kirim ulang pertanyaan Anda.";
}

function toAssistantMessage(response: ChatResponse): UiMessage {
  return {
    id: response.message_id,
    role: "assistant",
    content: response.answer,
    createdAt: response.created_at,
    status: response.status,
    citations: response.citations,
    sources: response.sources,
  };
}

export function useChat() {
  const [initialSession] = useState(() => localStorage.getItem(SESSION_KEY));
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [stage, setStage] = useState<PipelineStage | null>(null);
  const [pending, setPending] = useState(false);
  const [restoring, setRestoring] = useState(initialSession !== null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const pendingRef = useRef(false);

  useEffect(() => {
    const storedSession = initialSession;
    if (!storedSession) return;
    const controller = new AbortController();
    void getHistory(storedSession, controller.signal)
      .then((history) => {
        setSessionId(history.session_id);
        setMessages(
          history.messages.map((message) => ({
            id: message.id,
            role: message.role,
            content: message.content,
            createdAt: message.created_at,
            status: message.status ?? undefined,
            citations: message.citations,
            sources: message.sources,
          })),
        );
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        localStorage.removeItem(SESSION_KEY);
        setNotice("Sesi sebelumnya telah berakhir. Anda dapat memulai percakapan baru.");
      })
      .finally(() => setRestoring(false));
    return () => controller.abort();
  }, [initialSession]);

  const rememberSession = useCallback((value: string) => {
    setSessionId(value);
    localStorage.setItem(SESSION_KEY, value);
  }, []);

  const send = useCallback(
    async (rawMessage: string) => {
      const content = rawMessage.trim();
      if (!content || pendingRef.current) return;
      pendingRef.current = true;
      setPending(true);
      setError(null);
      setNotice(null);
      setStage("validating");
      const userMessage: UiMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        createdAt: new Date().toISOString(),
      };
      setMessages((current) => [...current, userMessage]);
      const controller = new AbortController();
      controllerRef.current = controller;
      try {
        const response = await streamChat(
          content,
          sessionId,
          { onSession: rememberSession, onStatus: setStage },
          controller.signal,
        );
        rememberSession(response.session_id);
        setMessages((current) => [...current, toAssistantMessage(response)]);
      } catch (caught: unknown) {
        if (!(caught instanceof DOMException && caught.name === "AbortError")) {
          setError(errorMessage(caught));
        }
      } finally {
        controllerRef.current = null;
        pendingRef.current = false;
        setPending(false);
        setStage(null);
      }
    },
    [rememberSession, sessionId],
  );

  const cancel = useCallback(() => controllerRef.current?.abort(), []);

  const newChat = useCallback(async () => {
    controllerRef.current?.abort();
    const previousSession = sessionId;
    setMessages([]);
    setSessionId(null);
    setError(null);
    setNotice(null);
    localStorage.removeItem(SESSION_KEY);
    if (previousSession) {
      try {
        await deleteSession(previousSession);
      } catch {
        setNotice("Percakapan lokal sudah dibersihkan, tetapi sesi server belum dapat dihapus.");
      }
    }
  }, [sessionId]);

  return {
    messages,
    stage,
    pending,
    restoring,
    error,
    notice,
    send,
    cancel,
    newChat,
  };
}
