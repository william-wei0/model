# Step8_unified_fusion.py
# === 统一融合模型（LSTM + MLP）实现 ===

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd

# === 设备设置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === 加载数据并对齐 ===

def load_and_align_data():
    seq_data = np.load(SEQ_DATA_PATH, allow_pickle=True)
    track_data = np.load(TRACK_DATA_PATH, allow_pickle=True)


    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, y_track, track_ids_track = track_data['X'], track_data['y'], track_data['track_ids']

    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        print("transposing...")
        X_seq = np.transpose(X_seq, (0, 2, 1))

    track_id_to_index = {
        tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,): i
        for i, tid in enumerate(track_ids_track)
    }

    X_seq_matched, X_track_matched, y_matched = [], [], []
    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            idx = track_id_to_index[key]
            X_seq_matched.append(X_seq[i])
            X_track_matched.append(X_track[idx])
            y_matched.append(y_seq[i])

    print(f"[DEBUG] Matched pairs: {len(X_seq_matched)}")
    return np.array(X_seq_matched), np.array(X_track_matched), np.array(y_matched)




# === 模型定义 ===
class UnifiedFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=9, hidden_size=64, batch_first=True, bidirectional=True)
        self.track_fc = nn.Sequential(
            nn.Linear(12, 64), nn.ReLU(), nn.Dropout(0.0)
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, x_seq, x_track):
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :]
        track_feat = self.track_fc(x_track)
        fused = torch.cat([lstm_feat, track_feat], dim=1)
        return self.fusion_fc(fused)


# move data adding and training inside

def Train_UnifiedFusionModel(seq_path, track_path, model_save_path, result_path):
    print("[STEP 1] Loading and aligning data...")
    X_seq, X_track, y = load_and_align_data(seq_path, track_path)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_seq_train, X_seq_test, X_track_train, X_track_test, y_train, y_test = train_test_split(
        X_seq, X_track, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)
    X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    X_track_test = torch.tensor(X_track_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_seq_train, X_track_train, y_train_tensor)
    test_dataset = TensorDataset(X_seq_test, X_track_test, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = UnifiedFusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("[STEP 2] Training unified fusion model...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_seq, batch_track, batch_y in train_loader:
            batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_seq, batch_track)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    print("[STEP 3] Evaluating...")
    model.eval()
    with torch.no_grad():
        logits = model(X_seq_test.to(device), X_track_test.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    print("[RESULT] Accuracy:", np.mean(preds == y_test))
    print(classification_report(y_test, preds, target_names=[str(cls) for cls in le.classes_]))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Unified Fusion Model Confusion Matrix")
    plt.savefig(result_path + "/confusion_matrix.png")
    plt.close()

    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

if  __name__ == "__main__":
    from Config import GENERATED_DIR, SEQ_LEN, MODEL_DIR, SEQ_RESULT_DIR
    SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
    TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
    MODEL_SAVE_PATH = f"{MODEL_DIR}/unified_fusion_model.pth"
    os.makedirs(SEQ_RESULT_DIR, exist_ok=True)
    
    Train_UnifiedFusionModel(SEQ_DATA_PATH, TRACK_DATA_PATH, MODEL_SAVE_PATH, SEQ_RESULT_DIR)