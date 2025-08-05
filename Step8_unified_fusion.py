import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd
from Config import EPOCHS, MODEL_DIR, RESULTS_DIR, BATCH_SIZE, FUSION_SIZE, TRACK_OUTPUT_SIZE
from Step3A_spotmodel import Attention
import random

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, "unified_model_best.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_align_data(seq_path, track_path):
    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

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


class UnifiedFusionModel(nn.Module):
    def __init__(self, seq_input_size, track_input_size, hidden_size=128, dropout=0.5):
        super().__init__()
        self.track_input_size = track_input_size

        self.lstm = nn.LSTM(input_size=seq_input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size*2, dropout)
        self.norm = nn.LayerNorm(hidden_size*2)

        if track_input_size > 0:
            self.track_fc = nn.Sequential(
                nn.Linear(track_input_size, TRACK_OUTPUT_SIZE),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # self.track_fc = nn.Sequential(
            #     nn.Linear(track_input_size, hidden_size),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(hidden_size, TRACK_OUTPUT_SIZE)
            # )

            # self.track_fc = nn.Sequential(
            #     nn.Linear(track_input_size, TRACK_OUTPUT_SIZE),
            # )

            self.use_track = True
        else:
            self.use_track = False

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + TRACK_OUTPUT_SIZE, FUSION_SIZE),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(FUSION_SIZE, 3)
        )


    def forward(self, x_seq, x_track):
        lstm_out, _ = self.lstm(x_seq)
        #lstm_out = self.norm(lstm_out)
        lstm_feat, attn_weights = self.attn(lstm_out)
        lstm_feat = F.layer_norm(lstm_feat, lstm_feat.shape[1:])
        

        if self.use_track:
            track_feat = self.track_fc(x_track)
            track_feat = F.layer_norm(track_feat, track_feat.shape[1:])
            fused = torch.cat([lstm_feat, track_feat], dim=1)
        else:
            fused = lstm_feat
        return self.fusion_fc(fused)
    

def get_weights(train_loader):
    num_classes = 3
    class_counts = torch.zeros(num_classes, dtype=torch.long).to(device)

    for batch_seq, batch_track, batch_y in train_loader:
        counts = torch.bincount(batch_y, minlength=num_classes).to(device)
        class_counts += counts
    
    print("Class counts:", class_counts)

    class_frequencies = class_counts.float() / class_counts.sum()
    print("Class frequencies:", class_counts.float() / class_counts.sum())

    weights = 1-class_frequencies
    #weights = torch.tensor([1.0, 1.0, 10.0]).to(device)
    print("weights:", weights)
    return weights

    

def Train_UnifiedFusionModel(seq_path, track_path, model_save_path, result_path,
                             seq_input_size=9, track_input_size=12, hidden_size=128, dropout=0.5, test_prefix="no_prefix"):
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    weights = get_weights(train_loader)

    model = UnifiedFusionModel(seq_input_size=seq_input_size, track_input_size=track_input_size,
                               hidden_size=hidden_size, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    print("[STEP 2] Training unified fusion model...")
    best_acc, early_stop = 0, 0
    entropy_total, entropy_count = 0.0, 0
    lowest_loss = 10000
    best_model = None
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_seq, batch_track, batch_y in train_loader:
            batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_seq, batch_track)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for batch_seq, batch_track, batch_y in test_loader:
                batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                probs = F.softmax(logits, dim=1)
                batch_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropy_total += batch_entropy.sum().item()
                entropy_count += batch_entropy.size(0)
                avg_entropy = entropy_total / entropy_count
                print(f"[VAL] Epoch {epoch+1} - Avg Prediction Entropy: {avg_entropy:.4f}")

                pred = logits.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)



        train_loss = train_loss / len(train_loader)
        val_loss /= len(test_loader)
        acc = correct / total
        print(scheduler.get_last_lr())
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(acc)

        print(f"Epoch {epoch + 1} | Loss = {train_loss:.4f} | Val Loss={val_loss:.4f} | Acc={acc:.4f}")

        if val_loss < lowest_loss:
            lowest_loss, best_model = val_loss, model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 40:
                print("Early stopping triggered.")
                break
                

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), UNIFIED_MODEL_PATH)

    lstm_weight_norm = sum(p.norm().item() for n, p in model.lstm.named_parameters() if 'weight' in n)
    track_weight_norm = sum(p.norm().item() for n, p in model.track_fc.named_parameters() if 'weight' in n)
    print("LSTM weight norm:", lstm_weight_norm)
    print("Track weight norm:", track_weight_norm)
    
    results_path = os.path.join(RESULTS_DIR, test_prefix)
    np.savez(f"{results_path}/training_logs_unified.npz", train_losses=train_losses, val_losses=val_losses, val_accuracies=val_accs)

    # Plot training curve
    print("[STEP 2] Drawing Loss Graph...")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{results_path}/loss_curve.png")
    plt.close()
    print("[STEP 2] Finished Drawing Loss Graph...")

    print("[STEP 2] Drawing Validation Accuracy Graph...")
    plt.plot(np.array(val_accs)*100)
    plt.title("Validation Accuracy (%)")
    plt.savefig(f"{results_path}/val_accuracy.png")
    plt.close()
    print("[STEP 2] Finished Drawing Validation Accuracy Graph...")
    

    print("[STEP 3] Evaluating...")
    model.eval()
    with torch.no_grad():
        logits = model(X_seq_test.to(device), X_track_test.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    acc = np.mean(preds == y_test)
    f1 = f1_score(y_test, preds, average="macro")

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    try:
        auc_value = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovo")
    except:
        auc_value = -1  # handle case when AUC cannot be computed
    print("[STEP 3] Drawing ROC Graph...")
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(result_path,"roc_curve.png"))
    plt.close()
    print("[STEP 3] Finished Drawing ROC Graph...")

    print("[RESULT] Accuracy:", acc)
    print(classification_report(y_test, preds, target_names=[str(cls) for cls in le.classes_]))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Unified Fusion Model Confusion Matrix")
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    plt.close()



    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc_value,
        "confusion_matrix": cm.tolist(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracy": val_accs 
    }

def Test_UnifiedFusionModel(seq_path, track_path, model_path, output_dir="Results"):
    print("[TEST] Loading external test dataset...")
    X_seq, X_track, y = load_and_align_data(seq_path, track_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    X_track_tensor = torch.tensor(X_track, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    model = UnifiedFusionModel(seq_input_size=X_seq.shape[2], track_input_size=X_track.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("[TEST] Running inference...")
    with torch.no_grad():
        logits = model(X_seq_tensor, X_track_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    acc = np.mean(preds == y_encoded)
    f1 = f1_score(y_encoded, preds, average="macro")
    y_bin = label_binarize(y_encoded, classes=np.unique(y_encoded))
    try:
        auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovo")
    except:
        auc = -1

    probs = np.array(probs)
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{SEQ_RESULT_DIR}/roc_curve.png")
    plt.close()
    

    print("[RESULT] External Test Accuracy:", acc)
    print(classification_report(y_encoded, preds, target_names=[str(cls) for cls in le.classes_]))

    # Save confusion matrix
    cm = confusion_matrix(y_encoded, preds)
    os.makedirs(output_dir, exist_ok=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (External Test Set)")
    plt.savefig(os.path.join(output_dir, "external_confusion_matrix.png"))
    plt.close()



    # Save predictions
    df_result = pd.DataFrame({
        "True": le.inverse_transform(y_encoded),
        "Pred": le.inverse_transform(preds)
    })
    df_result.to_csv(os.path.join(output_dir, "external_predictions.csv"), index=False)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    from Config import GENERATED_DIR, SEQ_LEN, MODEL_DIR, SEQ_RESULT_DIR
    SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
    TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
    MODEL_SAVE_PATH = f"{MODEL_DIR}/unified_fusion_model.pth"
    os.makedirs(SEQ_RESULT_DIR, exist_ok=True)
    Train_UnifiedFusionModel(SEQ_DATA_PATH, TRACK_DATA_PATH, MODEL_SAVE_PATH, SEQ_RESULT_DIR)

    TEST_SEQ_DATA_PATH = f"{GENERATED_DIR}/cart_test_trajectory_{SEQ_LEN}.npz"
    TEST_TRACK_DATA_PATH = f"{GENERATED_DIR}/cart_test_track.npz"
    MODEL_PATH = os.path.join(MODEL_DIR, "unified_model_best.pth")

    Test_UnifiedFusionModel(TEST_SEQ_DATA_PATH, TEST_TRACK_DATA_PATH, MODEL_PATH, output_dir=SEQ_RESULT_DIR)