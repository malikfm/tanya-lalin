import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

export const ChatInput = ({ onSend, disabled }: ChatInputProps) => {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 items-end">
      <Textarea
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Tanyakan sesuatu tentang hukum lalu lintas Indonesia..."
        disabled={disabled}
        className="min-h-[52px] max-h-32 resize-none rounded-2xl border-2 focus:border-primary transition-colors"
        rows={1}
      />
      <Button
        type="submit"
        disabled={disabled || !input.trim()}
        size="icon"
        className="h-[52px] w-[52px] rounded-2xl shrink-0"
      >
        <Send className="h-5 w-5" />
      </Button>
    </form>
  );
};
