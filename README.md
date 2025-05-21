# Deception Detection Project

A machine learning system for detecting deception in conversational text data. The system uses BERT embeddings combined with metadata features to identify potentially deceptive messages.

## Project Structure

```
deception_detection_project/
│
├── frontend/                 # Next.js web application
│   ├── app/                  # Pages and routing
│   ├── components/           # UI components
│   ├── lib/                  # Utility functions
│   └── ...                   # Next.js config files
│
├── backend/
│   ├── app.py                # FastAPI app to handle requests
│   ├── model/                # Trained model and assets
│   ├── preprocessing/        # Preprocessing utilities
│   ├── utils/                # Inference logic
│   └── requirements.txt      # Backend dependencies
│
├── training/
│   ├── train.py              # Training script
│   ├── data/                 # Training data
│   └── save_model.py         # Model saving logic
│
└── README.md                 # Project documentation
```

## Features

- **AI-powered deception detection**: Uses a fine-tuned BERT model with metadata features
- **Conversation analysis**: Analyze conversations to detect potentially deceptive messages
- **Interactive visualization**: View and filter conversations with deception insights
- **API integration**: Backend API for real-time deception detection

## Technical Details

The system consists of:

1. **Backend**:
   - FastAPI server providing deception detection APIs
   - Fine-tuned BERT language model for text analysis
   - Metadata feature processing and extraction

2. **Frontend**:
   - Next.js web application with TypeScript
   - Modern UI with intuitive conversation visualization
   - Interactive filtering and insights

3. **Machine Learning**:
   - BERT-based model for contextual text understanding
   - Metadata features for enhanced performance
   - Threshold calibration for optimal detection

## Getting Started

### Backend Setup

1. Create a Python environment:
   ```bash
   cd backend
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the API server:
   ```bash
   uvicorn app:app --reload
   ```

### Frontend Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

### Training a Model

1. Prepare your training data in CSV format in `training/data/`
2. Run the training script:
   ```bash
   cd training
   python train.py
   ```

3. Save the trained model to the backend:
   ```bash
   python save_model.py
   ```

## Usage

1. Open the web application (default: http://localhost:3000)
2. Upload a CSV file with conversation data
3. View the deception analysis results
4. Filter and explore the conversations

## License

This project is licensed under the MIT License - see the LICENSE file for details. 