import { cn } from "@/lib/utils";
import { ChevronDown, ChevronUp, FileText } from "lucide-react";
import { useState } from "react";

export interface RetrievedChunk {
  source: string;
  article_number: number | null;
  paragraph_number: number | null;
  chunk_type: string;
  text: string;
  similarity_score: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  retrievedChunks?: RetrievedChunk[];
}

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isUser = message.role === "user";
  const [showSources, setShowSources] = useState(false);
  const hasChunks = message.retrievedChunks && message.retrievedChunks.length > 0;

  const formatArticleRef = (chunk: RetrievedChunk) => {
    if (chunk.paragraph_number) {
      return `Pasal ${chunk.article_number} ayat (${chunk.paragraph_number})`;
    }
    return `Pasal ${chunk.article_number}`;
  };

  return (
    <div
      className={cn(
        "flex w-full mb-4 animate-in fade-in-50 slide-in-from-bottom-3 duration-300",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 shadow-sm",
          isUser
            ? "bg-[hsl(var(--message-user))] text-primary-foreground rounded-br-sm"
            : "bg-[hsl(var(--message-bot))] text-card-foreground border border-border rounded-bl-sm"
        )}
      >
        <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
        </p>

        {/* Sources toggle for assistant messages */}
        {!isUser && hasChunks && (
          <div className="mt-3 pt-2 border-t border-border/50">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <FileText className="h-3 w-3" />
              <span>{message.retrievedChunks!.length} sumber hukum</span>
              {showSources ? (
                <ChevronUp className="h-3 w-3" />
              ) : (
                <ChevronDown className="h-3 w-3" />
              )}
            </button>

            {showSources && (
              <div className="mt-2 space-y-2">
                {message.retrievedChunks!.map((chunk, index) => (
                  <div
                    key={index}
                    className="text-xs bg-background/50 rounded-lg p-2 border border-border/30"
                  >
                    <div className="flex items-center mb-1">
                      <span className="font-medium text-primary">
                        {formatArticleRef(chunk)}
                      </span>
                    </div>
                    <p className="text-muted-foreground line-clamp-3">
                      {chunk.text}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <span className="text-xs opacity-70 mt-2 block">
          {message.timestamp.toLocaleTimeString("id-ID", {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </span>
      </div>
    </div>
  );
};
