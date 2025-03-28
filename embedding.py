import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
from trainer import run
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Subset



# -------------------------
# Shared Classifier & Dataset Classes
# -------------------------
class ProteinClassifier(nn.Module):
    def __init__(self, embedding_size = 1280, num_classes= 4): # ProtBERT: 1024, ESM: 320, ESM-650: 1280
        """
        pretrained_model: The pre-trained protein transformer (e.g., ProtBert)
        hidden_size: Size of the embedding from the pre-trained model (e.g., 1024 for ProtBert)
        num_classes: Number of output classes (4 in our case)
        fine_tune: If False, freeze the base model parameters
        
        The classifier head takes the embedding from the [CLS] token (first token) and 
        passes it through a couple of fully connected layers to predict the 4 classes.
        """
        super().__init__()
        
        """
        # A simple classifier head on top of the [CLS] token embedding
        # Use case: protbert, protxlnet, protalbert
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        """ 
        # (backup: MLP version for ESM2-320, ESM2-650)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    
    def forward(self, embeddings):
        # Get the outputs from the pre-trained model
        # Here, we use the embedding corresponding to the CLS token.
        # outputs = self.pretrained_model(input_ids=input_ids,
        #                                attention_mask=attention_mask)
        # For BERT-like models, the first token is the [CLS] token
        #cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        #logits = self.classifier(cls_embedding)  # [batch_size, num_classes]
        logits = self.classifier(embeddings)
        return logits
    
class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, embeddings_file):
        """
        Loads precomputed embeddings and labels from a file.
        embeddings_file: Path to the .pt file containing embeddings.
        """
        data = torch.load(embeddings_file)
        self.embeddings = data["embeddings"]
        self.labels = data.get("labels") # If labels exist, load them; othersie set to None
    
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.embeddings[idx], self.labels[idx]
        else:
            return self.embeddings[idx]

def train_classifier(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device="cuda"):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    count_matrix = np.zeros((4, 4))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} Starting...")

        # Training Phase
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        for batch_idx, (embeddings, labels) in enumerate(train_loader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = (train_correct / total) * 100 
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        y_true_list = []
        y_pred_list = []
        y_prob_list = []
        count_matrix = np.zeros((4, 4)) 
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                logits = model(embeddings)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                val_total += labels.size(0)
                
                # Store labels, predictions, and probabilities
                y_true = labels.cpu().numpy()
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_prob = torch.softmax(logits, dim=1).cpu().numpy()

                y_true_list.append(y_true)
                y_pred_list.append(y_pred)
                y_prob_list.append(y_prob)

                count_matrix += confusion_matrix(y_true, y_pred, labels=np.arange(4))
        val_loss /= len(val_loader)
        val_acc = (val_correct / val_total) * 100
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc) 

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    y_true_final = np.array(y_true_list)
    y_pred_final = np.array(y_pred_list)
    y_prob_final = np.array(y_prob_list)

    return model, (train_losses, train_accuracies, val_losses, val_accuracies, count_matrix, y_true_final, y_pred_final, y_prob_final)

def compute_and_save_embeddings(dataset, model, tokenizer, save_path="precomputed_embeddings.pt",
                                batch_size=16, device="cuda", pooling_mode="cls"):

    model.to(device)
    model.eval()
    
    # 1) Prepare a data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 2) Figure out how to fetch labels (if they exist)
    if isinstance(dataset, Subset):
        full_dataset = dataset.dataset
        has_labels = hasattr(full_dataset, "labels") and full_dataset.labels is not None
        # Pre-gather the labels for the subset
        if has_labels:
            labels_list = [full_dataset.labels[i] for i in dataset.indices]
        else:
            labels_list = None
    else:
        # If it's a direct ProteinDataset
        has_labels = (dataset.labels is not None)
        if has_labels:
            labels_list = dataset.labels
        else:
            labels_list = None

    # 3) Loop over batches and compute embeddings
    embeddings_list = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if has_labels:
                input_ids, attention_mask, labels = batch
                all_labels.append(labels)
            else:
                input_ids, attention_mask = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if pooling_mode == "cls":
                # Take [CLS] embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            else:
                # Mean-pooling
                cls_embedding = torch.mean(outputs.last_hidden_state, dim=1)

            embeddings_list.append(cls_embedding.cpu())

    # 4) Concatenate
    all_embeddings = torch.cat(embeddings_list, dim=0)

    if has_labels:
        all_labels = torch.cat(all_labels, dim=0)
        torch.save({"X": all_embeddings, "y": all_labels}, save_path)
    else:
        torch.save({"X": all_embeddings}, save_path)

    print(f"Saved embeddings to {save_path}")


 
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels=None, tokenizer=None, max_length=2000):
        """
        sequences: List of raw protein sequences (strings)
        labels: List or array of integer labels (0,1,2,3)
        tokenizer: Pre-trained tokenizer (e.g., from ProtBert)
        max_length: Maximum tokenized sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)
    
    def preprocess_sequence(self, seq):
        # For ProtBert, we should add spaces between amino acids
        # e.g., "MKLFWLLFTIGFCWA" -> "M K L F W L L F T I G F C W A"
        return " ".join(list(seq))
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = self.preprocess_sequence(seq)
        
        # Tokenize the sequence
        encoded = self.tokenizer(seq,
                                 add_special_tokens=True,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=self.max_length,
                                 return_tensors="pt")
        # Squeeze to remove extra dimension and return input_ids and attention_mask
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        if self.labels is not None:
            label = self.labels[idx]
            return input_ids, attention_mask, label
        else:
            return input_ids, attention_mask



def experiment_esm_mlp_kfold(X, y, num_classes=4, n_splits=5, batch_size=32, 
                             num_epochs=100, lr=1e-3, device='cuda'):
    """
    X: Tensor of shape (N, 1280) [ESM-Large embeddings for all samples]
    y: Tensor of shape (N,) [integer labels]
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store per-fold results
    fold_metrics = {
        "best_val_acc": [],
        "mean_val_acc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
        "conf_matrices": []  # We'll store each fold's confusion matrix here
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n===== FOLD {fold_idx+1} / {n_splits} =====")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx], y[val_idx]

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset   = TensorDataset(X_val,   y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # Initialize MLP for ESM embeddings (dimension=1280)
        model = ProteinClassifier(embedding_size=1280, num_classes=num_classes).to(device)
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        all_val_accs = []

        # Training loop: train for num_epochs and compute validation accuracy each epoch
        for epoch in range(num_epochs):
            # --- Training ---
            model.train()
            running_loss, running_correct, total_samples = 0, 0, 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)  # shape = (batch_size, num_classes)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(batch_x)
                running_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total_samples += len(batch_x)

            train_loss = running_loss / total_samples
            train_acc = running_correct / total_samples

            # --- Validation: compute loss and accuracy, but DO NOT accumulate predictions here ---
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * len(batch_x)
                    val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    val_total += len(batch_x)
            val_loss /= val_total
            val_acc = val_correct / val_total
            all_val_accs.append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} "
                  f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} "
                  f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # --- After training, perform a final evaluation on the validation set once ---
        y_true_all, y_pred_all, y_prob_all = [], [], []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                y_true_all.extend(batch_y.cpu().numpy())
                y_pred_all.extend(logits.argmax(dim=1).cpu().numpy())
                y_prob_all.extend(torch.softmax(logits, dim=1).cpu().numpy())

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_prob_all = np.array(y_prob_all)

        # Compute confusion matrix for this fold
        cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(num_classes))
        fold_metrics["conf_matrices"].append(cm)
        
        # Compute per-fold metrics
        fold_metrics["best_val_acc"].append(best_val_acc)
        fold_metrics["mean_val_acc"].append(np.mean(all_val_accs))
        fold_metrics["f1"].append(f1_score(y_true_all, y_pred_all, average="macro"))
        fold_metrics["precision"].append(precision_score(y_true_all, y_pred_all, average="macro"))
        fold_metrics["recall"].append(recall_score(y_true_all, y_pred_all, average="macro"))
        fold_metrics["roc_auc"].append(roc_auc_score(y_true_all, y_prob_all, multi_class="ovr"))

        print(f"Fold {fold_idx+1} Results: "
              f"Best Val Acc = {best_val_acc:.3f}, Mean Val Acc = {np.mean(all_val_accs):.3f}, "
              f"F1 = {fold_metrics['f1'][-1]:.3f}, Precision = {fold_metrics['precision'][-1]:.3f}, "
              f"Recall = {fold_metrics['recall'][-1]:.3f}, ROC-AUC = {fold_metrics['roc_auc'][-1]:.3f}")

    # Compute the element-wise mean confusion matrix across all folds
    fold_metrics["conf_matrices"] = np.mean(fold_metrics["conf_matrices"], axis=0)

    # Compute and print overall mean across folds for the metrics
    for metric in ["best_val_acc", "mean_val_acc", "f1", "precision", "recall", "roc_auc"]:
        fold_metrics[metric] = np.mean(fold_metrics[metric])
    
    print("\n===== Overall 5-Fold Results =====")
    for metric, value in fold_metrics.items():
        if metric != "conf_matrices":  # Avoid printing the matrix
            print(f"{metric}: {value:.3f}")

    return fold_metrics



def train_on_full_data_and_test_esm(
    train_embeddings_path,
    test_embeddings_path,
    batch_size=32,
    num_classes=4,
    num_epoch=100,
    lr=1e-3,
    device="cuda"
):
    """
    1) Train on full dataset with precomputed ESM embeddings.
    2) Predict labels for the unlabeled test set.
    3) Save predictions & probabilities in a DataFrame.

    Args:
      train_embeddings_path: Path to precomputed full train embeddings (e.g. "all_esm_embeddings.pt")
      test_embeddings_path: Path to precomputed test embeddings (e.g. "test_esm_embeddings.pt")
      batch_size: Training batch size
      num_classes: Number of output classes
      num_epoch: Training epochs
      lr: Learning rate
      device: "cuda" or "cpu"

    Returns:
      trained_model: Trained ESM+MLP model
      train_stats: (train_losses, train_accs)
      df_predictions: DataFrame with [Class Probabilities + Predicted Label]
    """
    # --------------------
    # Load full ESM embeddings
    # --------------------
    data = torch.load(train_embeddings_path)
    X_train = data["embeddings"]  # shape = (N, 1280)
    y_train = data["labels"]  # shape = (N,)

    print(f"[INFO] Loaded X_train = {X_train.shape}, y_train = {y_train.shape} from {train_embeddings_path}")

    # Create dataset & dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   # --------------------
    # Train MLP on Full Data
    # --------------------
    model = ProteinClassifier(embedding_size=1280, num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []

    model.train()
    for epoch in range(num_epoch):
        running_loss, running_correct, total_samples = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_x)
            preds = logits.argmax(dim=1)
            running_correct += (preds == batch_y).sum().item()
            total_samples += len(batch_x)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.3f}")

    # --------------------
    # Load Precomputed Test ESM Embeddings
    # --------------------
    print(f"[INFO] Loading test embeddings from {test_embeddings_path}...")
    data_test = torch.load(test_embeddings_path)
    X_test = data_test["X"]  # shape (M, 320)

    print(f"[INFO] Loaded test embeddings: X_test = {X_test.shape}")

    # --------------------
    # Predict on Test Set (No Labels)
    # --------------------
    model.eval()
    y_probs_list = []
    y_pred_list = []

    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)  # shape (batch_size, num_classes)

            y_probs_list.append(probs.cpu())
            y_pred_list.append(torch.argmax(probs, dim=1).cpu())

    y_probs = torch.cat(y_probs_list, dim=0).numpy()  # shape=(M, num_classes)
    y_preds = torch.cat(y_pred_list, dim=0).numpy()

    # --------------------
    # Build DataFrame for Test Predictions
    # --------------------
    columns = [f"Class{i}_prob" for i in range(num_classes)]
    df = pd.DataFrame(y_probs, columns=columns)
    df["Prediction"] = y_preds  # Add final predicted class
    
    torch.save(model.state_dict(), "trained_esm_mlp.pth")

    return model, (train_losses, train_accs), df
