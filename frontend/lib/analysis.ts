import type { Conversation, Message, Player } from "@/types/types"
import { mockAnalyzeMessage } from "./api"

/**
 * Processes CSV data into conversation objects
 * In a production environment, this would typically be handled by the backend
 */
export function analyzeConversations(csvData: Record<string, string>[]): Conversation[] {
  // Group by conversation
  const conversationGroups: Record<string, any[]> = {}

  csvData.forEach((row) => {
    const convId = row.conversation_id || `conv_${Math.random().toString(36).substring(2, 9)}`
    if (!conversationGroups[convId]) {
      conversationGroups[convId] = []
    }
    conversationGroups[convId].push(row)
  })

  // Convert to conversation objects
  return Object.entries(conversationGroups).map(([id, rows], index) => {
    // Extract player info from first row
    const firstRow = rows[0]
    const player1: Player = {
      name: firstRow.player1_name || `Player ${index * 2 + 1}`,
      country: firstRow.player1_country || "Italy",
    }

    const player2: Player = {
      name: firstRow.player2_name || `Player ${index * 2 + 2}`,
      country: firstRow.player2_country || "Germany",
    }

    // Create messages with front-end analysis (when backend is not available)
    const messages: Message[] = rows.map((row, msgIndex) => {
      const isPlayer1 = row.sender === "player1" || 
                       (row.sender === undefined && msgIndex % 2 === 0)
      
      const text = row.message || row.text || `Message ${msgIndex + 1}`
      
      // Use pre-existing deception score or analyze on the client
      let deceptionScore = undefined
      let reasoning = undefined
      let confidence = undefined
      
      if (row.deception_score !== undefined) {
        // Use pre-existing score if available
        deceptionScore = Number.parseFloat(row.deception_score)
        reasoning = deceptionScore > 0.7 ? row.reasoning || "Suspicious language patterns detected" : undefined
        confidence = deceptionScore > 0.7 ? Number.parseInt(row.confidence) || Math.floor(Math.random() * 30) + 70 : undefined
      } else {
        // Analyze on the client side
        const analysis = mockAnalyzeMessage(text)
        deceptionScore = analysis.deceptionScore
        reasoning = analysis.reasoning
        confidence = reasoning ? Math.floor(deceptionScore * 100) : undefined
      }

      return {
        id: `${id}_msg_${msgIndex}`,
        sender: isPlayer1 ? player1 : player2,
        text,
        timestamp: row.timestamp || `${10 + msgIndex}:${String(Math.floor(Math.random() * 60)).padStart(2, "0")}`,
        deceptionScore,
        reasoning,
        confidence,
      }
    })

    // Determine if conversation is deceptive (any message with high score or pre-set flag)
    const isDeceptive = rows[0].is_deceptive === "true" || 
                       messages.some((m) => m.deceptionScore > 0.7)

    return {
      id,
      player1,
      player2,
      session: Number.parseInt(firstRow.session) || Math.floor(Math.random() * 3) + 1,
      round: Number.parseInt(firstRow.round) || Math.floor(Math.random() * 3) + 1,
      isDeceptive,
      messages,
    }
  })
}

/**
 * Extract metadata features from a message for the backend API
 */
export function extractMessageMetadata(text: string, position: number = 0, convoLength: number = 1) {
  // Basic features
  const messageLength = text.length
  const wordCount = text.trim().split(/\s+/).length
  const questionCount = (text.match(/\?/g) || []).length
  const exclamationCount = (text.match(/\!/g) || []).length
  
  // Check for markers of uncertainty and certainty
  const uncertaintyMarkers = [
    "maybe", "perhaps", "possibly", "might", "could", "not sure", 
    "guess", "think", "suspect", "seem", "appeared", "unsure", "doubt"
  ]
  const certaintyMarkers = [
    "definitely", "certainly", "absolutely", "always", "never", 
    "undoubtedly", "clearly", "obviously", "must", "sure", "know"
  ]
  
  const textLower = text.toLowerCase()
  const hasUncertainty = uncertaintyMarkers.some(marker => textLower.includes(marker)) ? 1 : 0
  const hasCertainty = certaintyMarkers.some(marker => textLower.includes(marker)) ? 1 : 0
  
  // Position features
  const positionRatio = convoLength > 0 ? position / convoLength : 0
  
  return {
    message_length: messageLength,
    word_count: wordCount,
    question_count: questionCount,
    exclamation_count: exclamationCount,
    has_uncertainty: hasUncertainty,
    has_certainty: hasCertainty,
    conversation_length: convoLength,
    msg_position_in_convo: position,
    position_ratio: positionRatio
  }
}
