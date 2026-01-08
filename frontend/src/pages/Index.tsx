import { useState, useRef, useEffect } from "react";
import { ChatMessage, Message, RetrievedChunk } from "@/components/ChatMessage";
import { ChatInput } from "@/components/ChatInput";
import { useToast } from "@/hooks/use-toast";
import { Scale, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

const SESSION_STORAGE_KEY = "tanya-lalin-session-id";
const API_BASE_URL = import.meta.env.VITE_API_URL || "";

interface ChatResponse {
  session_id: string;
  query: string;
  response: string;
  retrieved_chunks: RetrievedChunk[];
  timestamp: string;
}

const Index = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Load session from localStorage on mount
  useEffect(() => {
    const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
    if (storedSessionId) {
      setSessionId(storedSessionId);
      loadSessionHistory(storedSessionId);
    }
  }, []);

  const loadSessionHistory = async (sid: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat/${sid}/history`);
      if (response.ok) {
        const data = await response.json();
        const loadedMessages: Message[] = data.messages.map((msg: {
          id: string;
          role: "user" | "assistant";
          content: string;
          retrieved_chunks: RetrievedChunk[];
          created_at: string;
        }) => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.created_at),
          retrievedChunks: msg.retrieved_chunks || [],
        }));
        setMessages(loadedMessages);
      } else {
        // Session not found or expired, clear it
        localStorage.removeItem(SESSION_STORAGE_KEY);
        setSessionId(null);
      }
    } catch (error) {
      console.error("Failed to load session history:", error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleNewChat = () => {
    setMessages([]);
    setSessionId(null);
    localStorage.removeItem(SESSION_STORAGE_KEY);
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: content,
          top_k: 5,
          min_similarity: 0.3,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();

      // Save session ID
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        localStorage.setItem(SESSION_STORAGE_KEY, data.session_id);
      }

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || "Maaf, tidak ada respon dari server.",
        timestamp: new Date(),
        retrievedChunks: data.retrieved_chunks || [],
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      toast({
        title: "Error",
        description: "Gagal mengirim pesan. Pastikan server berjalan.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[image:var(--gradient-bg)]">
      {/* Header */}
      <header className="border-b bg-card/80 backdrop-blur-sm sticky top-0 z-10 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary rounded-xl">
              <Scale className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">Tanya Lalin</h1>
              <p className="text-sm text-muted-foreground">Asisten Hukum Lalu Lintas Indonesia</p>
            </div>
          </div>
          {messages.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleNewChat}
              className="gap-2"
            >
              <RotateCcw className="h-4 w-4" />
              Chat Baru
            </Button>
          )}
        </div>
      </header>

      {/* Chat Messages */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <div className="p-4 bg-primary/10 rounded-full mb-4">
                <Scale className="h-12 w-12 text-primary" />
              </div>
              <h2 className="text-2xl font-bold mb-2">Selamat Datang!</h2>
              <p className="text-muted-foreground max-w-md mb-6">
                Tanyakan apa saja tentang hukum lalu lintas Indonesia berdasarkan UU No. 22 Tahun 2009.
              </p>
              <div className="grid gap-2 text-left max-w-md">
                <p className="text-sm text-muted-foreground">Contoh pertanyaan:</p>
                <div className="space-y-2">
                  {[
                    "Apakah menerobos lampu merah melanggar aturan?",
                    "Apa sanksi jika tidak memakai helm?",
                    "Apakah bahu jalan boleh dilalui saat macet?",
                  ].map((example, i) => (
                    <button
                      key={i}
                      onClick={() => handleSendMessage(example)}
                      className="block w-full text-left text-sm p-3 bg-card/50 hover:bg-card border border-border rounded-lg transition-colors"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-card border border-border rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>
      </main>

      {/* Input Area */}
      <footer className="border-t bg-card/80 backdrop-blur-sm sticky bottom-0 shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <ChatInput onSend={handleSendMessage} disabled={isLoading} />
          <p className="text-xs text-center text-muted-foreground mt-2">
            Berdasarkan UU No. 22 Tahun 2009 tentang Lalu Lintas dan Angkutan Jalan
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
