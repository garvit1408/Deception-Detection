import os
import numpy as np
import re
from typing import List, Dict, Any, Union

# Constants
METADATA_FEATURES = [
    "message_length",
    "word_count",
    "question_count",
    "exclamation_count",
    "has_uncertainty",
    "has_certainty",
    "conversation_length",
    "msg_position_in_convo",
    "position_ratio",
]

# Optional features that might be present
OPTIONAL_FEATURES = ["sender_is_player", "prev_msg_truthful", "game_stage"]

# Basic threshold for mock model
THRESHOLD = 0.3

def extract_metadata(metadata_list: List[Dict[str, Any]]) -> np.ndarray:
    """Mock metadata extraction that returns the data as is"""
    # This is a simplification for the demo
    return np.array(metadata_list)

def generate_reasoning(text: str, probability: float) -> str:
    """Generate explanatory reasoning for high-probability deceptive texts"""
    if probability < 0.7:
        return None
    
    # Simple heuristics-based reasoning generator
    reasons = []
    
    if probability > 0.9:
        confidence = "Very high"
    elif probability > 0.8:
        confidence = "High"
    else:
        confidence = "Moderate"
    
    # Check for common deception indicators
    if "?" in text and text.count("?") >= 2:
        reasons.append("Excessive questioning may indicate deflection")
    
    if "!" in text:
        reasons.append("Emphatic statements may indicate overcompensation")
    
    if any(word in text.lower() for word in ["honestly", "truthfully", "believe me", "trust me"]):
        reasons.append("Overuse of trustworthiness assertions")
    
    if any(word in text.lower() for word in ["never", "always", "all", "none", "every", "absolutely"]):
        reasons.append("Use of absolute terms may indicate exaggeration")
        
    if len(text.split()) < 5:
        reasons.append("Unusually short message may indicate evasiveness")
    
    if not reasons:
        reasons.append("Suspicious language patterns detected")
        
    return f"{confidence} confidence: {', '.join(reasons)}"

def predict_deception(texts: List[str], metadata: List[Dict[str, Any]]) -> Dict[str, Union[List[float], List[bool]]]:
    """
    Mock deception prediction for demonstration purposes
    
    Args:
        texts: List of text messages to analyze
        metadata: List of dictionaries containing metadata for each message
        
    Returns:
        Dictionary with predictions, probabilities and reasoning
    """
    # Mock prediction logic based on simple heuristics
    probabilities = []
    
    for text in texts:
        text_lower = text.lower()
        
        # Base score (slightly random for variety)
        score = 0.1 + np.random.random() * 0.2
        
        # Increase score for suspicious patterns
        if "?" in text and text.count("?") >= 2:
            score += 0.2
            
        if "!" in text:
            score += 0.15
            
        if any(word in text_lower for word in ["honestly", "truthfully", "believe me", "trust me"]):
            score += 0.3
            
        if any(word in text_lower for word in ["never", "always", "all", "none", "every", "absolutely"]):
            score += 0.25
            
        if len(text.split()) < 5:
            score += 0.2
        
        # For demonstration, make certain words highly indicative
        deceptive_phrases = [
            "absolutely", "certainly", "definitely", "trust me", "believe me",
            "swear", "never", "always", "exactly", "100%"
        ]
        for phrase in deceptive_phrases:
            if phrase in text_lower:
                score += 0.2
                
        honest_phrases = [
            "i think", "possibly", "maybe", "perhaps", "not sure",
            "might be", "could be", "consider", "i feel", "in my opinion"
        ]
        for phrase in honest_phrases:
            if phrase in text_lower:
                score -= 0.1
        
        # Cap at 0.95
        score = min(score, 0.95)
        score = max(score, 0.05)  # Ensure minimum score of 0.05
        
        probabilities.append(score)
    
    # Convert to numpy arrays for consistency
    probabilities = np.array(probabilities)
    
    # Apply threshold to get predictions
    predictions = [prob >= THRESHOLD for prob in probabilities]
    
    # Generate reasoning for highly probable deceptive messages
    reasoning = [generate_reasoning(text, prob) for text, prob in zip(texts, probabilities)]
    
    return {
        "predictions": predictions,
        "probabilities": probabilities.tolist(),
        "reasoning": reasoning
    } 