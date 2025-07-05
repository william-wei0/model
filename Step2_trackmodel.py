import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from Config import *


# ==== Configuration ====
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ==== Load Data ====
data = np.load(f"{GENERATED_DIR}/track_dataset.npz")
X = data['X']
y = data['y']

print("Loaded track dataset:", X.shape, y.shape)

track_features = [f"feature_{i}" for i in range(X.shape[1])]

# ==== Encode Labels ====
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==== Train/Val/Test Split ====
total_size = len(X)
val_size = int(total_size * VAL_RATIO)
test_size = int(total_size * TEST_RATIO)
train_size = total_size - val_size - test_size

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# ==== Define Model ====
class TrackNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = TrackNet(input_dim=X.shape[1])
class_counts = np.bincount(y_encoded)
class_weights = torch.tensor(len(y_encoded) / (len(np.unique(y_encoded)) * class_counts), dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== EvaluationLoop ====
def evaluate(loader):
    model.eval()
    total, correct = 0, 0
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())
            y_scores.extend(probs.numpy())

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    acc = correct / total
    return acc, y_true, y_pred, np.array(y_scores)

if __name__ == "__main__":
    print("Start training...")
    train_losses = []
    val_accuracies = []
    test_accuracies = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, _, _, _ = evaluate(val_loader)
        test_acc, _, _, _ = evaluate(test_loader)
        train_losses.append(total_loss)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch:3d}/{EPOCHS} - Train Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ==== Final Evaluation ====
    test_acc, y_true, y_pred, y_scores = evaluate(test_loader)
    print("\nFinal Test Accuracy:", test_acc)
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # ==== Loss & Accuracy Plot ====
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color='tab:blue')
    ax1.plot(range(1, EPOCHS + 1), train_losses, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color='tab:red')
    ax2.plot(range(1, EPOCHS + 1), val_accuracies, color='orange', label='Val Acc')
    ax2.plot(range(1, EPOCHS + 1), test_accuracies, color='tab:red', label='Test Acc')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title("Training Loss and Accuracy")
    plt.savefig(f"{TRACK_RESULT_DIR}/track_loss_accuracy_combined.png")

    # ==== Confusion Matrix ====
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{TRACK_RESULT_DIR}/track_confusion_matrix.png")

    # ==== ROC Curve ====
    n_classes = len(le.classes_)
    plt.figure(figsize=(7,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve([1 if y==i else 0 for y in y_true], y_scores[:, i])
        auc = roc_auc_score([1 if y==i else 0 for y in y_true], y_scores[:, i])
        plt.plot(fpr, tpr, label=f"Class {le.classes_[i]} vs Rest (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (1-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{TRACK_RESULT_DIR}/track_roc_curve.png")

    # ==== Dimensionality Reduction ====
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()

    # ==== PCA on input features ====
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_np)
    plt.figure()
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=le.inverse_transform(y_np), palette="Set1")
    plt.title("PCA Visualization")
    plt.savefig(f"{TRACK_RESULT_DIR}/track_pca.png")
    # ==== PCA on model output ====
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor).numpy()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(logits)
    y_np = y_tensor.numpy()
    plt.figure()
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], 
                    hue=le.inverse_transform(y_np), palette="Set1", alpha=0.7)
    plt.title("PCA Visualization")
    plt.savefig(f"{TRACK_RESULT_DIR}/track_pca.png")

# ==== t-SNE and UMAP ====
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X_np)
    plt.figure()
    sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=le.inverse_transform(y_np), palette="Set2")
    plt.title("t-SNE Visualization")
    plt.savefig(f"{TRACK_RESULT_DIR}/track_tsne.png")

    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(X_np)
    plt.figure()
    sns.scatterplot(x=umap_result[:,0], y=umap_result[:,1], hue=le.inverse_transform(y_np), palette="Set3")
    plt.title("UMAP Visualization")
    plt.savefig(f"{TRACK_RESULT_DIR}/track_umap.png")

    # ==== Save Model ====
    torch.save(model.state_dict(), f"{MODEL_DIR}/track_model.pth")
    print(f"Model saved to {MODEL_DIR}/track_model.pth")
