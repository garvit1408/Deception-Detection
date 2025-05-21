import os
import shutil
import torch
import joblib
from transformers import BertTokenizer
import argparse

def save_model(
    model_path: str, 
    tokenizer_path: str, 
    scaler_path: str, 
    threshold_path: str, 
    destination_dir: str
):
    """
    Save the trained model and its associated files to the backend model directory
    
    Args:
        model_path: Path to the PyTorch model weights
        tokenizer_path: Path to the tokenizer directory
        scaler_path: Path to the metadata scaler
        threshold_path: Path to the threshold file
        destination_dir: Destination directory for saved model files
    """
    os.makedirs(destination_dir, exist_ok=True)
    
    # Create paths for destination files
    dest_model_path = os.path.join(destination_dir, "model.pt")
    dest_tokenizer_dir = os.path.join(destination_dir, "tokenizer")
    dest_scaler_path = os.path.join(destination_dir, "metadata_scaler.pkl")
    dest_threshold_path = os.path.join(destination_dir, "threshold.txt")
    
    # Copy model weights
    print(f"Copying model from {model_path} to {dest_model_path}")
    shutil.copy2(model_path, dest_model_path)
    
    # Copy tokenizer
    print(f"Copying tokenizer from {tokenizer_path} to {dest_tokenizer_dir}")
    os.makedirs(dest_tokenizer_dir, exist_ok=True)
    
    # Handle tokenizer differently - it's a directory
    if os.path.isdir(tokenizer_path):
        for item in os.listdir(tokenizer_path):
            s = os.path.join(tokenizer_path, item)
            d = os.path.join(dest_tokenizer_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
            else:
                shutil.copytree(s, d, dirs_exist_ok=True)
    
    # Copy scaler
    print(f"Copying scaler from {scaler_path} to {dest_scaler_path}")
    shutil.copy2(scaler_path, dest_scaler_path)
    
    # Copy threshold
    print(f"Copying threshold from {threshold_path} to {dest_threshold_path}")
    shutil.copy2(threshold_path, dest_threshold_path)
    
    print(f"Model and associated files successfully saved to {destination_dir}")

def main():
    parser = argparse.ArgumentParser(description="Save trained model files to backend directory")
    
    parser.add_argument(
        "--model-path", 
        default="results/model.pt",
        help="Path to the trained model weights"
    )
    
    parser.add_argument(
        "--tokenizer-path", 
        default="results/tokenizer",
        help="Path to the tokenizer directory"
    )
    
    parser.add_argument(
        "--scaler-path", 
        default="results/metadata_scaler.pkl",
        help="Path to the metadata scaler"
    )
    
    parser.add_argument(
        "--threshold-path", 
        default="results/threshold.txt",
        help="Path to the classification threshold file"
    )
    
    parser.add_argument(
        "--destination", 
        default="../backend/model",
        help="Destination directory for saved model files"
    )
    
    args = parser.parse_args()
    
    # Save the model
    save_model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        scaler_path=args.scaler_path,
        threshold_path=args.threshold_path,
        destination_dir=args.destination
    )

if __name__ == "__main__":
    main() 