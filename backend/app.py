from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import traceback

from utils.predict import predict_deception

app = FastAPI(title="Deception Detection API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}
    
class MessageInput(BaseModel):
    messages: List[TextInput]

@app.get("/")
def read_root():
    return {"status": "Deception Detection API is running"}

@app.post("/predict")
async def predict_single(input: TextInput):
    """Predict deception for a single message with metadata"""
    try:
        result = predict_deception(
            texts=[input.text], 
            metadata=[input.metadata]
        )
        return {
            "prediction": bool(result["predictions"][0]),
            "probability": float(result["probabilities"][0]),
            "reasoning": result["reasoning"][0] if "reasoning" in result else None
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(input: MessageInput):
    """Predict deception for a batch of messages with metadata"""
    try:
        texts = [item.text for item in input.messages]
        metadata = [item.metadata for item in input.messages]
        
        result = predict_deception(texts=texts, metadata=metadata)
        
        return {
            "predictions": [bool(p) for p in result["predictions"]],
            "probabilities": [float(p) for p in result["probabilities"]],
            "reasoning": result.get("reasoning", [None] * len(texts))
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-conversation")
async def analyze_conversation(file: UploadFile = File(...)):
    """Analyze a full conversation from CSV file"""
    try:
        # For demo purposes, we're creating a mock conversation response
        # In a real scenario, we'd parse the CSV and analyze each message
        
        file_contents = await file.read()
        file_text = file_contents.decode('utf-8')
        
        # Simple message extraction from CSV
        lines = file_text.split('\n')
        if len(lines) <= 1:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
            
        # Extract messages (assuming 1st column contains messages)
        header = lines[0].split(',')
        message_idx = 0  # Default to first column
        
        # Try to find message/text column
        for i, col in enumerate(header):
            if "message" in col.lower() or "text" in col.lower():
                message_idx = i
                break
                
        # Extract messages from rows, skipping header
        messages = []
        for i, line in enumerate(lines[1:]):
            if not line.strip():  # Skip empty lines
                continue
                
            cols = line.split(',')
            if len(cols) <= message_idx:
                continue  # Skip if row doesn't have enough columns
                
            message_text = cols[message_idx].strip()
            if not message_text:
                continue
                
            # Analyze with mock model
            result = predict_deception([message_text], [{}])
            
            messages.append({
                "text": message_text,
                "sender": {
                    "name": f"Player {i % 2 + 1}",
                    "country": "Italy" if i % 2 == 0 else "Germany"
                },
                "timestamp": f"{10 + i}:{(i * 5) % 60:02d}",
                "deceptionScore": float(result["probabilities"][0]),
                "reasoning": result["reasoning"][0],
                "confidence": int(float(result["probabilities"][0]) * 100) if result["reasoning"][0] else None
            })
        
        # Determine if conversation is deceptive (any message with high score)
        is_deceptive = any(msg["deceptionScore"] > 0.7 for msg in messages)
        
        conversation = {
            "id": f"conv_{file.filename.split('.')[0]}",
            "player1": {"name": "Player 1", "country": "Italy"},
            "player2": {"name": "Player 2", "country": "Germany"},
            "session": 1,
            "round": 1,
            "isDeceptive": is_deceptive,
            "messages": messages
        }
        
        return {"conversation": conversation}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 