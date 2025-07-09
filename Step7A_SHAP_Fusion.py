import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
from Config import *
from Step2A_trackmodel import TrackNet
from Step3A_spotmodel import BiLSTMAttnModel

# === Config ===
SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
TRACK_MODEL_PATH = f"{MODEL_DIR}/track_model.pth"
SPOT_MODEL_PATH = f"{MODEL_DIR}/model_best_{SEQ_LEN}.pth"
FUSION_MODEL_PATH = f"{MODEL_DIR}/fusion_model.pth"

class FusionNet(nn.Module):
    def __init__(self, input_dim=149, hidden_dim=64, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# === Load Data ===
print("[STEP 1] Loading data...")
seq_data = np.load(SEQ_DATA_PATH, allow_pickle=True)
track_data = np.load(TRACK_DATA_PATH, allow_pickle=True)
X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
X_track, y_track, track_ids_track = track_data['X'], track_data['y'], track_data['track_ids']

# === Match IDs and Extract Raw + Mid-level Features ===
def to_key(tid):
    return tuple(tid) if isinstance(tid, (list, np.ndarray)) else (tid,)

seq_map = {to_key(tid): i for i, tid in enumerate(track_ids_seq)}
track_map = {to_key(tid): i for i, tid in enumerate(track_ids_track)}
shared_keys = set(seq_map.keys()) & set(track_map.keys())

print(f"[STEP 2] Extracting features... total shared keys: {len(shared_keys)}")

raw_seq_feats, raw_track_feats = [], []
mid_seq_feats, mid_track_feats = [], []
labels = []

# Load models to extract mid-level features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_seq = BiLSTMAttnModel(input_dim=FEATURE_LEN, hidden_dim=64, output_dim=3, dropout=0.3).to(device)
model_seq.load_state_dict(torch.load(SPOT_MODEL_PATH, map_location=device))
model_seq.eval()

model_track = TrackNet(input_dim=X_track.shape[1], num_classes=3).to(device)
model_track.load_state_dict(torch.load(TRACK_MODEL_PATH, map_location=device))
model_track.eval()

for key in shared_keys:
    i_seq = seq_map[key]
    i_track = track_map[key]
    x_seq = X_seq[i_seq]              # shape (T, 9)
    x_track = X_track[i_track]       # shape (12,)

    raw_seq_feats.append(x_seq.mean(axis=0))
    raw_track_feats.append(x_track)

    with torch.no_grad():
        x_seq_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
        ctx = model_seq.attn(model_seq.lstm(x_seq_tensor)[0])[0]  # (1, 128)
        mid_seq = model_seq.fc[0](ctx).squeeze(0).cpu().numpy()   # (64,)

        x_track_tensor = torch.tensor(x_track, dtype=torch.float32).unsqueeze(0).to(device)
        mid_track = model_track.net[0](x_track_tensor).squeeze(0).cpu().numpy()  # (64,)

    mid_seq_feats.append(mid_seq)
    mid_track_feats.append(mid_track)
    labels.append(y_track[i_track])

X_orig = np.hstack([raw_seq_feats, raw_track_feats])        # shape (N, 21)
X_mid = np.hstack([mid_seq_feats, mid_track_feats])         # shape (N, 128)
X_fused = np.hstack([X_orig, X_mid])                        # shape (N, 149)
y = np.array(labels)

# === Train/Test Split ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_fused, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# === Train FusionNet ===
print("[STEP 3] Training fusion model...")
fusion_model = FusionNet().to(device)
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

fusion_model.train()
for epoch in range(20):
    optimizer.zero_grad()
    outputs = fusion_model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1:2d}: Loss = {loss.item():.4f}")

# === Evaluation ===
fusion_model.eval()
with torch.no_grad():
    preds = fusion_model(X_test_tensor.to(device))
    preds_label = torch.argmax(preds, dim=1).cpu().numpy()

acc = accuracy_score(y_test, preds_label)
print("[RESULT] Fusion Model Accuracy:", acc)

# Ensure class names are string
class_names = [str(c) for c in le.classes_]
print(classification_report(y_test, preds_label, target_names=class_names))

cm = confusion_matrix(y_test, preds_label)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Fusion Model Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{FUSION_RESULT_DIR}/fusion_confusion_matrix_v2.png")

# === Save Model ===
torch.save(fusion_model.state_dict(), FUSION_MODEL_PATH)
print(f"Fusion model saved to {FUSION_MODEL_PATH}")

# === SHAP Explanation on First 21 dims only ===
print("[STEP 4] SHAP analysis...")
explainer = shap.GradientExplainer(fusion_model, X_train_tensor.to(device))
shap_values = explainer.shap_values(X_test_tensor.to(device))

if isinstance(shap_values, list):
    print("[DEBUG] shap_values is list, stacking along last axis")
    shap_tensor = np.stack(shap_values, axis=-1)  # (N, F, C)
    print("[DEBUG] stacked shap_tensor.shape =", shap_tensor.shape)
    shap_tensor = shap_tensor.transpose(1, 2, 0)  # (F, C, N)
    print("[DEBUG] transposed shap_tensor.shape =", shap_tensor.shape)
else:
    print("[DEBUG] shap_values originally ndarray, shape:", shap_values.shape)
    shap_tensor = shap_values.transpose(0, 2, 1)  # assume (N, F, C) → (F, C, N)
    print("[DEBUG] reshaped shap_tensor.shape =", shap_tensor.shape)

print("[DEBUG] final shap_tensor.shape =", shap_tensor.shape)

mean_shap = np.abs(shap_tensor).mean(axis=(1, 2))  # shape (F,)
print("[DEBUG] mean_shap.shape:", mean_shap.shape)

from Config import features, track_features
feature_names = features + track_features  # 9 + 12 = 21
print("[DEBUG] feature_names length:", len(feature_names))

mean_shap_orig = mean_shap[:len(feature_names)]
print("[DEBUG] mean_shap_orig shape:", mean_shap_orig.shape)

if len(mean_shap_orig) != len(feature_names):
    raise ValueError(f"[ERROR] SHAP feature length mismatch: mean_shap_orig={mean_shap_orig.shape}, feature_names={len(feature_names)}")

shap_df = pd.DataFrame({"Feature": feature_names, "Importance": mean_shap_orig})
shap_df = shap_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=shap_df, x="Importance", y="Feature")
plt.title("SHAP Feature Importance - Original Features Only")
plt.tight_layout()
plt.savefig(f"{FUSION_RESULT_DIR}/fusion_shap_bar.png")
print("SHAP feature importance saved.")
