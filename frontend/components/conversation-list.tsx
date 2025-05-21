"use client"

import type { Conversation } from "@/types/types"
import { cn } from "@/lib/utils"

interface ConversationListProps {
  conversations: Conversation[]
  selectedId: string
  onSelect: (id: string) => void
  filters: {
    country: string
    modelConfidence: number
    showOnlyDeceptive: boolean
  }
}

export default function ConversationList({ conversations, selectedId, onSelect, filters }: ConversationListProps) {
  // Apply filters
  const filteredConversations = conversations.filter((conv) => {
    if (filters.showOnlyDeceptive && !conv.isDeceptive) return false
    if (filters.country !== "All") {
      const hasCountry = conv.player1.country === filters.country || conv.player2.country === filters.country
      if (!hasCountry) return false
    }
    return true
  })

  return (
    <div className="bg-white rounded-md shadow-card overflow-hidden">
      <div className="p-4 font-medium text-lg border-b bg-gradient-card text-insight">Conversation List</div>
      <div className="divide-y">
        {filteredConversations.map((conversation) => (
          <div
            key={conversation.id}
            onClick={() => onSelect(conversation.id)}
            className={cn(
              "p-4 cursor-pointer hover:bg-secondary transition-colors",
              selectedId === conversation.id && "bg-accent",
              conversation.isDeceptive && "bg-deceptive-bg hover:bg-deceptive-bg/80",
            )}
          >
            <div className="font-medium text-insight-dark">
              {conversation.player1.name} ({conversation.player1.country}) vs {conversation.player2.name}{" "}
              {conversation.player2.country ? `(${conversation.player2.country})` : ""}
            </div>
            {conversation.isDeceptive && <div className="text-deceptive-dark font-medium">Deceptive</div>}
            <div className="text-sm text-neutral mt-1">
              Session {conversation.session} / Round {conversation.round}
            </div>
          </div>
        ))}

        {filteredConversations.length === 0 && (
          <div className="p-4 text-center text-neutral">No conversations match the current filters</div>
        )}
      </div>
    </div>
  )
}
