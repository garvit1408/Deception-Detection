import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Union, Tuple

# Download required NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('words')

# Constants for linguistic markers
UNCERTAINTY_MARKERS = [
    "maybe", "perhaps", "possibly", "might", "could", "not sure", 
    "guess", "think", "suspect", "seem", "appeared", "unsure", "doubt",
    "wonder", "question", "uncertain", "unclear", "confused"
]

CERTAINTY_MARKERS = [
    "definitely", "certainly", "absolutely", "always", "never", 
    "undoubtedly", "clearly", "obviously", "must", "sure", "know", 
    "positive", "convinced", "without a doubt", "no doubt", "truly"
]

def clean_text(text: str) -> str:
    """Basic text cleaning for messages"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_metadata_features(message: str, position: int = 0, convo_length: int = 1) -> Dict[str, float]:
    """
    Extract metadata features from a message
    
    Args:
        message: The text message
        position: Position of the message in the conversation (0-indexed)
        convo_length: Total length of the conversation
        
    Returns:
        Dictionary of metadata features
    """
    if not isinstance(message, str):
        message = str(message)
    
    # Tokenize the message
    tokens = word_tokenize(message.lower())
    
    # Basic features
    message_length = len(message)
    word_count = len(tokens)
    question_count = message.count('?')
    exclamation_count = message.count('!')
    
    # Check for markers of uncertainty and certainty
    has_uncertainty = any(marker in message.lower() for marker in UNCERTAINTY_MARKERS)
    has_certainty = any(marker in message.lower() for marker in CERTAINTY_MARKERS)
    
    # Position features
    position_ratio = (position + 1) / convo_length if convo_length > 0 else 0
    
    return {
        "message_length": message_length,
        "word_count": word_count,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "has_uncertainty": 1 if has_uncertainty else 0,
        "has_certainty": 1 if has_certainty else 0,
        "conversation_length": convo_length,
        "msg_position_in_convo": position,
        "position_ratio": position_ratio
    }

def preprocess_single_message(
    message: str, 
    metadata: Dict[str, Any] = None,
    position: int = 0, 
    convo_length: int = 1
) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess a single message for model inference
    
    Args:
        message: Raw message text
        metadata: Optional pre-existing metadata
        position: Position of message in conversation
        convo_length: Total length of conversation
        
    Returns:
        Tuple of (cleaned_message, metadata_features)
    """
    # Clean the message text
    cleaned_message = clean_text(message)
    
    # Extract metadata features
    extracted_metadata = extract_metadata_features(
        message=cleaned_message,
        position=position,
        convo_length=convo_length
    )
    
    # Combine with any pre-existing metadata
    if metadata:
        extracted_metadata.update(metadata)
    
    return cleaned_message, extracted_metadata

def preprocess_conversation(
    messages: List[str], 
    metadata_list: List[Dict[str, Any]] = None
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Preprocess a full conversation for model inference
    
    Args:
        messages: List of raw message texts
        metadata_list: Optional list of pre-existing metadata for each message
        
    Returns:
        Tuple of (cleaned_messages, metadata_features)
    """
    convo_length = len(messages)
    cleaned_messages = []
    all_metadata = []
    
    for i, message in enumerate(messages):
        # Get pre-existing metadata if available
        existing_metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
        
        # Preprocess the message
        cleaned, metadata = preprocess_single_message(
            message=message,
            metadata=existing_metadata,
            position=i,
            convo_length=convo_length
        )
        
        cleaned_messages.append(cleaned)
        all_metadata.append(metadata)
        
        # Add previous message truthfulness if available
        if i > 0 and "is_truthful" in all_metadata[i-1]:
            all_metadata[i]["prev_msg_truthful"] = all_metadata[i-1]["is_truthful"]
    
    return cleaned_messages, all_metadata

def preprocess_dataframe(df: pd.DataFrame, text_column: str = "message") -> pd.DataFrame:
    """
    Preprocess a dataframe with conversation data for training
    
    Args:
        df: Input dataframe with messages
        text_column: Name of the column containing message text
        
    Returns:
        DataFrame with preprocessed features
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Clean messages
    result_df["cleaned_message"] = result_df[text_column].apply(clean_text)
    
    # Group by conversation ID if it exists
    if "conversation_id" in result_df.columns:
        grouped = result_df.groupby("conversation_id")
        
        # Process each conversation
        for convo_id, group in grouped:
            convo_length = len(group)
            
            # Update metadata for each message
            for i, (idx, row) in enumerate(group.iterrows()):
                # Extract metadata
                metadata = extract_metadata_features(
                    message=row["cleaned_message"],
                    position=i,
                    convo_length=convo_length
                )
                
                # Add metadata to the dataframe
                for key, value in metadata.items():
                    result_df.at[idx, key] = value
                
                # Add previous message truthfulness if available
                if i > 0 and "is_truthful" in result_df.columns:
                    prev_idx = group.iloc[i-1].name
                    result_df.at[idx, "prev_msg_truthful"] = result_df.at[prev_idx, "is_truthful"]
    else:
        # Process as a single conversation
        convo_length = len(result_df)
        
        # Create metadata features
        for i, (idx, row) in enumerate(result_df.iterrows()):
            # Extract metadata
            metadata = extract_metadata_features(
                message=row["cleaned_message"],
                position=i,
                convo_length=convo_length
            )
            
            # Add metadata to the dataframe
            for key, value in metadata.items():
                result_df.at[idx, key] = value
            
            # Add previous message truthfulness if available
            if i > 0 and "is_truthful" in result_df.columns:
                prev_idx = result_df.iloc[i-1].name
                result_df.at[idx, "prev_msg_truthful"] = result_df.at[prev_idx, "is_truthful"]
    
    return result_df 