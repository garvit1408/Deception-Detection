import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sklearn.metrics
import argparse

# Add parent directory to path to import preprocessing module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.preprocessing.preprocess import clean_text, extract_metadata_features

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the model architecture
class DeceptionBERTModel(nn.Module):
    def __init__(self, bert_model_name="distilbert-base-uncased", metadata_dim=9):
        super(DeceptionBERTModel, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dim = self.bert.config.hidden_size

        # Layers for processing metadata
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Combined layers
        self.combined_fc = nn.Sequential(
            nn.Linear(self.bert_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask, metadata):
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

        # Process metadata
        metadata_embedding = self.metadata_fc(metadata)

        # Combine embeddings
        combined_embedding = torch.cat((cls_embedding, metadata_embedding), dim=1)

        # Final prediction
        output = self.combined_fc(combined_embedding)

        return output


# Custom dataset class
class DeceptionDataset(Dataset):
    def __init__(self, texts, metadata, labels=None, tokenizer=None, max_length=64):
        self.texts = texts
        self.metadata = metadata
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get input_ids and attention_mask
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Get metadata
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float)

        # Create item dict
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "metadata": metadata,
        }

        # Add label if available
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def load_processed_data(data_dir="data"):
    """Load the processed data from CSV files"""
    print("Loading processed data...")

    train_df = pd.read_csv(f"{data_dir}/processed_train.csv")
    val_df = pd.read_csv(f"{data_dir}/processed_val.csv")
    test_df = pd.read_csv(f"{data_dir}/processed_test.csv")

    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Check class balance in training data
    train_truthful = train_df["is_truthful"].sum()
    train_deceptive = len(train_df) - train_truthful
    print(
        f"Training data class balance: {train_truthful} truthful, {train_deceptive} deceptive"
    )
    print(f"Truthful percentage: {train_truthful / len(train_df) * 100:.2f}%")

    return train_df, val_df, test_df


def get_metadata_features(df):
    """Get metadata features from the dataframe"""
    metadata_features = [
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

    # Add sender features if they exist
    if "sender_is_player" in df.columns:
        metadata_features.append("sender_is_player")

    # Add previous message features if they exist
    if "prev_msg_truthful" in df.columns:
        metadata_features.append("prev_msg_truthful")

    # Add game stage features if they exist
    if "game_stage" in df.columns:
        metadata_features.append("game_stage")

    return metadata_features


def create_datasets(train_df, val_df, test_df, tokenizer, metadata_features):
    """Create datasets for training, validation, and testing"""
    print("Creating datasets...")

    # Preprocess text data
    train_texts = train_df["cleaned_message"].fillna("").values
    val_texts = val_df["cleaned_message"].fillna("").values
    test_texts = test_df["cleaned_message"].fillna("").values

    # Get labels
    train_labels = train_df["is_truthful"].values
    val_labels = val_df["is_truthful"].values
    test_labels = test_df["is_truthful"].values

    # Process categorical and boolean features to ensure they're numeric
    for col in metadata_features:
        for df in [train_df, val_df, test_df]:
            if col in df.columns:
                # Convert boolean columns
                if (
                    df[col].dtype == bool
                    or df[col].dtype == "object"
                    and df[col].isin([True, False, "True", "False"]).all()
                ):
                    df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0})
                # Convert other non-numeric columns to numeric if needed
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                    except:
                        # If conversion fails, create dummies or map to integers
                        print(f"Converting categorical column {col} to numeric")
                        categories = df[col].astype("category").cat.categories
                        df[col] = df[col].astype("category").cat.codes

    # Get metadata
    print(f"Using {len(metadata_features)} metadata features: {metadata_features}")
    train_metadata = train_df[metadata_features].values
    val_metadata = val_df[metadata_features].values
    test_metadata = test_df[metadata_features].values

    # Scale metadata
    metadata_scaler = StandardScaler()
    train_metadata = metadata_scaler.fit_transform(train_metadata)
    val_metadata = metadata_scaler.transform(val_metadata)
    test_metadata = metadata_scaler.transform(test_metadata)

    # Save scaler for inference
    os.makedirs("../backend/model", exist_ok=True)
    joblib.dump(metadata_scaler, "../backend/model/metadata_scaler.pkl")

    # Create datasets
    train_dataset = DeceptionDataset(
        train_texts, train_metadata, train_labels, tokenizer
    )
    val_dataset = DeceptionDataset(val_texts, val_metadata, val_labels, tokenizer)
    test_dataset = DeceptionDataset(test_texts, test_metadata, test_labels, tokenizer)

    return train_dataset, val_dataset, test_dataset


def train_model(
    model,
    train_dataset,
    val_dataset,
    device,
    batch_size=16,
    num_epochs=3,
    learning_rate=2e-5,
):
    """Train the model with early stopping and learning curves"""
    print("Training BERT model...")

    # Calculate class weights for balanced training
    train_labels = np.array([item["label"].item() for item in train_dataset])
    truthful_count = np.sum(train_labels)
    deceptive_count = len(train_labels) - truthful_count

    # Fixed weight for deceptive class (higher due to imbalance)
    weight_for_0 = 3.0  # Deceptive class
    weight_for_1 = 1.0  # Truthful class
    
    print(
        f"Class weights: [{weight_for_0:.4f}, {weight_for_1:.4f}] (deceptive, truthful)"
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=len(train_loader) * num_epochs
    )

    # Loss function with class weights
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight_for_0 / weight_for_1]).to(device)
    )

    # Training loop with early stopping
    best_val_f1 = 0.0
    patience = 2
    epochs_no_improve = 0

    # For tracking metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    print(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)")
        for batch in train_pbar:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, metadata)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update metrics
            train_loss += loss.item() * input_ids.size(0)
            train_total += input_ids.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": train_correct / train_total,
                }
            )

        # Calculate average training metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)")
        with torch.no_grad():
            for batch in val_pbar:
                # Move data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                metadata = batch["metadata"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                # Forward pass
                outputs = model(input_ids, attention_mask, metadata)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Update metrics
                val_loss += loss.item() * input_ids.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()

                # Collect predictions and labels for metrics
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        # Calculate average validation metrics
        val_loss = val_loss / len(val_dataset)
        val_preds = np.array(val_preds).flatten()
        val_labels_list = np.array(val_labels_list).flatten()
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_f1 = f1_score(val_labels_list, val_preds)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            # Save model
            torch.save(model.state_dict(), "../backend/model/model.pt")
            print(f"Model improved! Best F1: {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # Plot learning curves
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history["val_f1"], label="Validation")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/learning_curves.png")
    plt.close()

    # Load best model
    model.load_state_dict(torch.load("../backend/model/model.pt"))
    print(f"Loaded best model with validation F1: {best_val_f1:.4f}")

    return model


def evaluate_model(model, test_dataset, device, batch_size=16):
    """Evaluate the model on the test set"""
    print("Evaluating model on test set...")

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluation mode
    model.eval()
    test_preds = []
    test_probs = []
    test_labels_list = []

    # Set threshold for classification (lower than 0.5 to reduce truthful bias)
    threshold = 0.3
    print(f"Using classification threshold: {threshold}")

    # Save threshold for inference
    with open("../backend/model/threshold.txt", "w") as f:
        f.write(str(threshold))

    # Disable gradient computation
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            metadata = batch["metadata"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            # Forward pass
            outputs = model(input_ids, attention_mask, metadata)

            # Get probabilities and predictions
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).float()  # Using the custom threshold

            # Collect predictions, probabilities, and labels
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    test_preds = np.array(test_preds).flatten()
    test_probs = np.array(test_probs).flatten()
    test_labels_list = np.array(test_labels_list).flatten()

    # Calculate metrics
    metrics = calculate_metrics(test_labels_list, test_preds, test_probs, threshold)

    # Save predictions
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(
        {
            "prediction": test_preds,
            "probability": test_probs,
            "true_label": test_labels_list,
        }
    ).to_csv("results/predictions.csv", index=False)

    # Calibrate threshold and plot ROC curve
    calibrate_threshold(test_labels_list, test_probs)

    return metrics


def calculate_metrics(y_true, y_pred, y_prob, threshold=0.3):
    """Calculate metrics for model evaluation"""
    print("Calculating metrics...")

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Ensure the confusion matrix has the right shape (2x2)
    if cm.shape != (2, 2):
        if cm.shape == (1, 1):
            # Only one class in predictions and true values
            # Extend to a 2x2 matrix
            if y_true[0] == 1:
                # Only truthful examples
                cm = np.array([[0, 0], [0, cm[0, 0]]])
            else:
                # Only deceptive examples
                cm = np.array([[cm[0, 0], 0], [0, 0]])
        else:
            # Add missing rows/columns with zeros
            if cm.shape[0] == 1:
                if y_true[0] == 1:
                    cm = np.array([[0, 0], [0, cm[0, 0]]])
                else:
                    cm = np.array([[cm[0, 0], 0], [0, 0]])
            elif cm.shape[1] == 1:
                if y_pred[0] == 1:
                    cm = np.array([[0, 0], [cm[0, 0], 0]])
                else:
                    cm = np.array([[cm[0, 0], 0], [0, 0]])

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    balanced_acc = (specificity + sensitivity) / 2

    # Record results
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "balanced_accuracy": balanced_acc,
        "threshold": threshold,
        "confusion_matrix": cm,
    }

    # Print metrics
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Classification Threshold: {threshold}")
    print(f"Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Deceptive", "Truthful"],
        yticklabels=["Deceptive", "Truthful"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # Save metrics to a file
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(
            {
                "accuracy": float(acc),
                "balanced_accuracy": float(balanced_acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "specificity": float(specificity),
                "sensitivity": float(sensitivity),
                "threshold": float(threshold),
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=4,
        )

    return metrics


def calibrate_threshold(y_true, y_prob):
    """Find the best threshold and plot ROC curve"""
    print("Calibrating classification threshold...")

    # Calculate ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_prob)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    # Find the optimal threshold that maximizes the geometric mean of sensitivity and specificity
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    best_threshold = thresholds[ix]

    # Apply the best threshold
    y_pred_best = (y_prob >= best_threshold).astype(int)

    # Calculate confusion matrix
    cm_best = confusion_matrix(y_true, y_pred_best)

    # Calculate metrics with best threshold
    acc_best = accuracy_score(y_true, y_pred_best)
    precision_best = precision_score(y_true, y_pred_best, zero_division=0)
    recall_best = recall_score(y_true, y_pred_best, zero_division=0)
    f1_best = f1_score(y_true, y_pred_best, zero_division=0)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.scatter(
        fpr[ix],
        tpr[ix],
        marker="o",
        color="black",
        label=f"Best threshold = {best_threshold:.4f}",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig("results/roc_curve.png")
    plt.close()

    # Save calibration results
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"With best threshold - Accuracy: {acc_best:.4f}, F1: {f1_best:.4f}")
    print(f"Confusion matrix with best threshold:")
    print(cm_best)

    os.makedirs("results", exist_ok=True)
    with open("results/calibration.json", "w") as f:
        json.dump(
            {
                "best_threshold": float(best_threshold),
                "accuracy": float(acc_best),
                "precision": float(precision_best),
                "recall": float(recall_best),
                "f1": float(f1_best),
                "roc_auc": float(roc_auc),
                "confusion_matrix": cm_best.tolist(),
            },
            f,
            indent=4,
        )

    # Save best threshold for inference
    with open("../backend/model/threshold.txt", "w") as f:
        f.write(str(best_threshold))

    return best_threshold


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a deception detection model")
    
    parser.add_argument(
        "--data-dir", 
        default="data",
        help="Directory containing the processed data CSV files"
    )
    
    parser.add_argument(
        "--results-dir", 
        default="results",
        help="Directory to save results (plots, metrics, etc.)"
    )
    
    parser.add_argument(
        "--model-name", 
        default="distilbert-base-uncased",
        help="Name of the pretrained BERT model to use"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=16,
        help="Batch size for training (will be reduced for CPU training)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float,
        default=2e-5,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--balance-data", 
        action="store_true",
        help="Balance the training data by oversampling the minority class"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("../backend/model", exist_ok=True)
    
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Adjust batch size based on device
    batch_size = args.batch_size if device.type == "cuda" else 4
    print(f"Using batch size: {batch_size}")
    
    # Load data
    train_df, val_df, test_df = load_processed_data(args.data_dir)
    
    # Balance training data if requested
    if args.balance_data:
        print("Balancing training data...")
        truthful_df = train_df[train_df["is_truthful"] == 1]
        deceptive_df = train_df[train_df["is_truthful"] == 0]
        
        # Oversample the minority class
        if len(truthful_df) > len(deceptive_df):
            n_to_sample = len(truthful_df) - len(deceptive_df)
            oversampled = deceptive_df.sample(n_to_sample, replace=True, random_state=42)
            train_df = pd.concat([truthful_df, deceptive_df, oversampled])
        else:
            n_to_sample = len(deceptive_df) - len(truthful_df)
            oversampled = truthful_df.sample(n_to_sample, replace=True, random_state=42)
            train_df = pd.concat([truthful_df, deceptive_df, oversampled])
        
        # Shuffle the data
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Check new class balance
        train_truthful = train_df["is_truthful"].sum()
        train_deceptive = len(train_df) - train_truthful
        print(f"Balanced training data: {train_truthful} truthful, {train_deceptive} deceptive")
    
    # Initialize tokenizer
    print(f"Using {args.model_name} as base model")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Get metadata features
    metadata_features = get_metadata_features(train_df)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df, tokenizer, metadata_features
    )
    
    # Initialize model
    print("Initializing model...")
    model = DeceptionBERTModel(args.model_name, metadata_dim=len(metadata_features))
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    model = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        device, 
        batch_size=batch_size, 
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # Evaluate model
    print("Evaluating model on test set...")
    metrics = evaluate_model(model, test_dataset, device, batch_size)
    
    # Save model and tokenizer
    print("Saving model and tokenizer...")
    model_path = os.path.join(args.results_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    tokenizer_dir = os.path.join(args.results_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    
    print(f"Training completed. Results saved to {args.results_dir}")
    print(f"Use save_model.py to copy the model to the backend directory")

if __name__ == "__main__":
    main()