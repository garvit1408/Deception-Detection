import type { Conversation, Message } from "@/types/types";

// API configuration
const API_URL = typeof window !== 'undefined'
  ? window.location.origin.includes('localhost') 
    ? "http://localhost:8000" 
    : `${window.location.protocol}//${window.location.hostname}:8000`
  : "http://localhost:8000";

/**
 * Base API call helper function
 */
async function fetchAPI(endpoint: string, options = {}) {
  const url = `${API_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "An error occurred");
  }

  return response.json();
}

/**
 * Analyze a single message for deception
 */
export async function analyzeSingleMessage(text: string, metadata: any = {}) {
  return fetchAPI("/predict", {
    method: "POST",
    body: JSON.stringify({ text, metadata }),
  });
}

/**
 * Analyze a batch of messages for deception
 */
export async function analyzeBatch(messages: { text: string; metadata?: any }[]) {
  return fetchAPI("/predict-batch", {
    method: "POST",
    body: JSON.stringify({
      messages: messages.map(m => ({
        text: m.text,
        metadata: m.metadata || {}
      }))
    }),
  });
}

/**
 * Upload a CSV file for conversation analysis
 */
export async function uploadConversation(file: File): Promise<{ conversation: Conversation }> {
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API_URL}/analyze-conversation`;
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to analyze conversation");
  }

  return response.json();
}

/**
 * Mock function to simulate deception analysis on the frontend
 * This is used when the backend is not available
 */
export function mockAnalyzeMessage(text: string): { deceptionScore: number; reasoning?: string } {
  // Simple heuristic-based scoring
  let score = 0.1 + Math.random() * 0.2; // Base score between 0.1-0.3
  
  // Increase score for suspicious patterns
  if (text.includes("?") && text.split("?").length > 2) score += 0.2;
  if (text.includes("!")) score += 0.15;
  if (/\b(honestly|truthfully|believe me|trust me)\b/i.test(text)) score += 0.3;
  if (/\b(never|always|all|none|every|absolutely)\b/i.test(text)) score += 0.25;
  if (text.split(" ").length < 5) score += 0.2;
  
  // Cap at 0.95
  score = Math.min(score, 0.95);
  
  // Generate reasoning for highly deceptive messages
  let reasoning;
  if (score > 0.7) {
    const reasons = [];
    if (text.includes("?") && text.split("?").length > 2) reasons.push("Excessive questioning may indicate deflection");
    if (text.includes("!")) reasons.push("Emphatic statements may indicate overcompensation");
    if (/\b(honestly|truthfully|believe me|trust me)\b/i.test(text)) reasons.push("Overuse of trustworthiness assertions");
    if (/\b(never|always|all|none|every|absolutely)\b/i.test(text)) reasons.push("Use of absolute terms may indicate exaggeration");
    if (text.split(" ").length < 5) reasons.push("Unusually short message may indicate evasiveness");
    
    if (reasons.length > 0) {
      reasoning = reasons.join(", ");
    } else {
      reasoning = "Suspicious language patterns detected";
    }
  }
  
  return {
    deceptionScore: score,
    reasoning
  };
} 