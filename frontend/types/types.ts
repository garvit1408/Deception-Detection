export interface Player {
  name: string
  country: string | null
}

export interface Message {
  id: string
  sender: Player
  text: string
  timestamp: string
  deceptionScore: number
  reasoning?: string
  confidence?: number
}

export interface Conversation {
  id: string
  player1: Player
  player2: Player
  session: number
  round: number
  isDeceptive: boolean
  messages: Message[]
}
