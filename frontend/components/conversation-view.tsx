import type { Conversation, Message } from "@/types/types"
import { cn } from "@/lib/utils"
import { ItalyFlag, GermanyFlag } from "@/components/country-flags"

interface ConversationViewProps {
  conversation: Conversation | null
}

export default function ConversationView({ conversation }: ConversationViewProps) {
  if (!conversation) {
    return (
      <div className="bg-white rounded-md shadow-sm h-full flex items-center justify-center p-8">
        <p className="text-gray-500 text-center">Select a conversation to view</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-md shadow-card overflow-hidden h-full flex flex-col">
      <div className="p-4 font-medium text-lg border-b bg-gradient-card text-insight">Conversation</div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {conversation.messages.length === 0 ? (
          <p className="text-center text-neutral">No messages in this conversation</p>
        ) : (
          conversation.messages.map((message) => <MessageBubble key={message.id} message={message} />)
        )}
      </div>
    </div>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isItaly = message.sender.country === "Italy"
  const isGermany = message.sender.country === "Germany"
  const isHighDeception = message.deceptionScore > 0.7
  const isMediumDeception = message.deceptionScore > 0.3 && message.deceptionScore <= 0.7
  const isLowDeception = message.deceptionScore <= 0.3

  return (
    <div className="flex items-start gap-3">
      <div className="mt-1">
        {isItaly && <ItalyFlag />}
        {isGermany && <GermanyFlag />}
      </div>

      <div className="flex-1 space-y-1">
        <div
          className={cn(
            "p-3 rounded-lg inline-block max-w-[80%]",
            isHighDeception
              ? "bg-deceptive-bg border border-deceptive-light"
              : isMediumDeception
                ? "bg-amber-50 border border-amber-200"
                : "bg-gray-50 border border-gray-200",
          )}
        >
          <div className="text-insight-dark">{message.text}</div>

          {isHighDeception && message.reasoning && (
            <div className="mt-2 pt-2 border-t border-deceptive-light">
              <div className="text-deceptive-dark font-medium">High deceptive</div>
              <div className="text-sm">
                <div className="font-medium text-insight-dark">Model Reasoning</div>
                <div className="text-neutral-dark">{message.reasoning}</div>
                <div className="mt-1">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-neutral-dark">Confidence:</span>
                    <span className="text-deceptive-dark font-medium">{message.confidence}%</span>
                  </div>
                  <div className="confidence-indicator">
                    <div className="confidence-indicator-fill high" style={{ width: `${message.confidence}%` }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="text-xs text-neutral">{message.timestamp}</div>

        {(isMediumDeception || isLowDeception) && message.deceptionScore > 0 && (
          <div className="flex items-center gap-2">
            <div className="confidence-indicator w-16">
              <div
                className={cn("confidence-indicator-fill", isMediumDeception ? "medium" : "low")}
                style={{ width: `${message.deceptionScore * 100}%` }}
              ></div>
            </div>
            <div className="text-xs text-neutral">{Math.round(message.deceptionScore * 100)}%</div>
          </div>
        )}
      </div>

      {isHighDeception && <div className="text-right text-sm font-medium text-deceptive-dark">High</div>}
    </div>
  )
}
