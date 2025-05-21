"use client"

import { useState } from "react"
import ConversationList from "@/components/conversation-list"
import ConversationView from "@/components/conversation-view"
import InsightsPanel from "@/components/insights-panel"
import Header from "@/components/header"
import type { Conversation } from "@/types/types"
import { analyzeConversations } from "@/lib/analysis"
import { uploadConversation } from "@/lib/api"

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: "1",
      player1: { name: "Player 1", country: "Italy" },
      player2: { name: "Player 2", country: "Germany" },
      session: 1,
      round: 1,
      isDeceptive: false,
      messages: [
        {
          id: "1",
          sender: { name: "Player 1", country: "Italy" },
          text: "I have a auto.",
          timestamp: "10:05",
          deceptionScore: 0.1,
        },
        {
          id: "2",
          sender: { name: "Player 2", country: "Germany" },
          text: "I finished all of my tasks three hours ago.",
          timestamp: "10:07",
          deceptionScore: 0.8,
          reasoning: "Unlikely timing reported",
          confidence: 87,
        },
        {
          id: "3",
          sender: { name: "Player 1", country: "Italy" },
          text: "Should I get more.",
          timestamp: "10:10",
          deceptionScore: 0.2,
        },
        {
          id: "4",
          sender: { name: "Player 2", country: "Germany" },
          text: "Sure you next month?",
          timestamp: "10:30",
          deceptionScore: 0.17,
        },
      ],
    },
    {
      id: "2",
      player1: { name: "Player 2", country: "Italy" },
      player2: { name: "Player 2", country: "Germany" },
      session: 1,
      round: 1,
      isDeceptive: false,
      messages: [],
    },
    {
      id: "3",
      player1: { name: "Player 1", country: "Italy" },
      player2: { name: "Hing", country: null },
      session: 2,
      round: 1,
      isDeceptive: true,
      messages: [],
    },
    {
      id: "4",
      player1: { name: "Player 3", country: "Italy" },
      player2: { name: "Player 2", country: "Germany" },
      session: 1,
      round: 3,
      isDeceptive: false,
      messages: [],
    },
    {
      id: "5",
      player1: { name: "Player 4", country: "Italy" },
      player2: { name: "Player 2", country: "Germany" },
      session: 1,
      round: 3,
      isDeceptive: false,
      messages: [],
    },
    {
      id: "6",
      player1: { name: "Player 5", country: "Italy" },
      player2: { name: "Player 2", country: "Germany" },
      session: 1,
      round: 3,
      isDeceptive: false,
      messages: [],
    },
  ])

  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(conversations[0])
  const [insights, setInsights] = useState({
    deceptivePercentage: 33,
    suspiciousPhrases: ["Unlikely timing", "Contradictory statements"],
    confidenceHeatmap: [0.3, 0.1, 0.2, 0.1, 0.1, 0.4, 0.7],
  })
  const [filters, setFilters] = useState({
    country: "All",
    modelConfidence: 25,
    showOnlyDeceptive: false,
  })
  const [isLoading, setIsLoading] = useState(false)

  // Function to update insights based on a conversation
  const updateInsightsFromConversation = (conversation: Conversation) => {
    // Calculate deceptive percentage
    const deceptiveMessages = conversation.messages.filter(m => m.deceptionScore > 0.7).length;
    const totalMessages = conversation.messages.length;
    const percentage = totalMessages > 0 ? Math.round((deceptiveMessages / totalMessages) * 100) : 0;
    
    // Extract suspicious phrases
    const phrases: string[] = [];
    conversation.messages.forEach(msg => {
      if (msg.reasoning) {
        // Split reasoning into phrases and add to list
        const newPhrases = msg.reasoning.split(", ");
        newPhrases.forEach(phrase => {
          if (!phrases.includes(phrase) && phrases.length < 5) {
            phrases.push(phrase);
          }
        });
      }
    });
    
    // If no phrases were found, add default
    if (phrases.length === 0) {
      phrases.push("No specific patterns");
    }
    
    // Generate a heatmap from deception scores
    const heatmap = conversation.messages
      .map(m => m.deceptionScore)
      .slice(0, 7); // Take first 7 messages
    
    // Pad heatmap if needed
    while (heatmap.length < 7) {
      heatmap.push(0.1);
    }
    
    setInsights({
      deceptivePercentage: percentage,
      suspiciousPhrases: phrases,
      confidenceHeatmap: heatmap,
    });
  };

  const handleFileUpload = async (file: File) => {
    try {
      // Show loading indicator
      setIsLoading(true);
      
      try {
        // Try to use our backend API
        const response = await uploadConversation(file);
        if (response && response.conversation) {
          const newConversations = [response.conversation, ...conversations];
          setConversations(newConversations);
          setSelectedConversation(response.conversation);
          
          // Update insights based on new data
          updateInsightsFromConversation(response.conversation);
          
          alert(`Successfully processed conversation from CSV using backend API!`);
          return;
        }
      } catch (apiError) {
        console.error("Backend API not available, falling back to client-side processing", apiError);
        // Continue with client-side fallback processing below
      }
      
      // Fallback to client-side processing if backend API fails
      const formData = new FormData()
      formData.append("file", file)

      const text = await file.text()
      const lines = text.split("\n")
      const headers = lines[0].split(",")

      // Parse CSV (simplified for demo)
      const parsedData = lines.slice(1).map((line) => {
        const values = line.split(",")
        const row: Record<string, string> = {}
        headers.forEach((header, index) => {
          row[header.trim()] = values[index]?.trim() || ""
        })
        return row
      })

      // Convert to conversations
      const newConversations = analyzeConversations(parsedData)
      setConversations(newConversations)
      setSelectedConversation(newConversations[0] || null)

      // Update insights based on new data
      const deceptiveCount = newConversations.filter((c: Conversation) => c.isDeceptive).length
      const percentage = Math.round((deceptiveCount / newConversations.length) * 100)

      setInsights({
        deceptivePercentage: percentage,
        suspiciousPhrases: ["Detected from uploaded data", "Timing inconsistencies"],
        confidenceHeatmap: [0.4, 0.2, 0.3, 0.5, 0.1, 0.6, 0.8],
      })

      alert(`Successfully processed ${newConversations.length} conversations from CSV (client-side)!`)
    } catch (error) {
      console.error("Error processing file:", error)
      alert("Error processing file. Please check the format and try again.")
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <Header onFileUpload={handleFileUpload} />

      <div className="container mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div className="lg:col-span-3">
          <ConversationList
            conversations={conversations}
            selectedId={selectedConversation?.id || ""}
            onSelect={(id) => setSelectedConversation(conversations.find((c) => c.id === id) || null)}
            filters={filters}
          />
        </div>

        <div className="lg:col-span-6">
          <ConversationView conversation={selectedConversation} />
        </div>

        <div className="lg:col-span-3">
          <InsightsPanel insights={insights} filters={filters} onFilterChange={setFilters} />
        </div>
      </div>
    </main>
  )
}
