import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

from Step3A_spotmodel import BiLSTMAttnModel
from Step2A_trackmodel import TrackNet
from Config import *

# === Config ===
SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # GPU RAM too small

# === Load Data ===
def load_data():
    print("Loading data...")
    seq_data = np.load(SEQ_DATA_PATH, allow_pickle=True)
    X_seq, y_seq = seq_data['X'], seq_data['y']
    track_ids_seq = seq_data['track_ids']

    track_data = np.load(TRACK_DATA_PATH, allow_pickle=True)
    print("track_dataset.npz keys:", track_data.files)
    X_track, y_track = track_data['X'], track_data['y']
    track_ids_track = track_data['track_ids']

    print("[DEBUG] X_seq:", X_seq.shape, "y_seq:", y_seq.shape)
    print("[DEBUG] X_track:", X_track.shape, "y_track:", y_track.shape)
    print("[DEBUG] Sequence track_ids example:", track_ids_seq[0])
    print("[DEBUG] Track track_ids example:", track_ids_track[0])

    return X_seq, y_seq, track_ids_seq, X_track, y_track, track_ids_track


def build_track_id_to_label_map(track_ids, y):
    mapping = {}
    for i, tid in enumerate(track_ids):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        mapping[key] = str(y[i])  # force convert into string
    print(f"[DEBUG] Track ID label map built: {len(mapping)} entries")
    return mapping


# === Predict with both models ===
def predict_with_models(X_seq, X_track, track_ids_seq, track_ids_track, track_label_map):
    print("Loading models...")

    model_seq = BiLSTMAttnModel(input_dim=X_seq.shape[2], hidden_dim=64, output_dim=3, dropout=0.3).to(device)
    model_seq.load_state_dict(torch.load(f"{MODEL_DIR}/model_best_{SEQ_LEN}.pth",
                                         map_location=device, 
                                         weights_only=True))
    model_seq.eval()

    model_track = TrackNet(input_dim=X_track.shape[1], num_classes=3).to(device)
    model_track.load_state_dict(torch.load(f"{MODEL_DIR}/track_model.pth",
                                           map_location=device, 
                                           weights_only=True))
    model_track.eval()

    print("Generating predictions...")
    with torch.no_grad():
        pred_seq = F.softmax(model_seq(torch.tensor(X_seq, dtype=torch.float32
                                                    ).to(device))[0], dim=1).cpu().numpy()
        pred_track = F.softmax(model_track(torch.tensor(X_track, dtype=torch.float32
                                                        ).to(device)), dim=1).cpu().numpy()

    print("Mapping predictions to track IDs...")
    seq_map, track_map = {}, {}
    for i, tid in enumerate(track_ids_seq):
        seq_map[tuple(tid)] = pred_seq[i]
    for i, tid in enumerate(track_ids_track):
        track_map[tuple(tid)] = pred_track[i]

    matched_pred_seq, matched_pred_track, matched_labels = [], [], []
    shared_ids = set(seq_map.keys()) & set(track_map.keys()) & set(track_label_map.keys())
    print(f"[DEBUG] Shared track IDs found: {len(shared_ids)}")

    for tid in shared_ids:
        matched_pred_seq.append(seq_map[tid])
        matched_pred_track.append(track_map[tid])
        matched_labels.append(track_label_map[tid])

    print("Label distribution of matched set:", np.unique(matched_labels, return_counts=True))

    return np.array(matched_pred_seq), np.array(matched_pred_track), np.array(matched_labels)


# === Evaluate and Plot ===
def train_and_evaluate_fusion(pred_seq, pred_track, y_true, save_result=True):
    print("Training fusion classifier...")

    # all label converted to string，avoid reading float as continue number
    y_true = np.array(y_true).astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_true)
    print("[DEBUG] Detected classes:", le.classes_)
    print("Label distribution:", np.unique(y_encoded, return_counts=True))

    pred_seq_label = le.inverse_transform(np.argmax(pred_seq, axis=1)).astype(str)
    pred_track_label = le.inverse_transform(np.argmax(pred_track, axis=1)).astype(str)

    acc_seq = accuracy_score(y_true, pred_seq_label)
    acc_track = accuracy_score(y_true, pred_track_label)

    best_acc = 0
    best_w = 0
    best_pred = None
    acc_list = []
    weights = np.linspace(0, 1, 21)

    for w in weights:
        fusion_pred = w * pred_seq + (1 - w) * pred_track
        y_pred_encoded = np.argmax(fusion_pred, axis=1)
        y_pred_label = le.inverse_transform(y_pred_encoded).astype(str)

        acc = accuracy_score(y_true, y_pred_label)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_pred = y_pred_label

    print(f"[RESULT] Best Fusion Weight: {best_w:.2f} | Accuracy: {best_acc:.4f}")

    print("Classification Report:")
    print(classification_report(y_true, best_pred))

    cm = confusion_matrix(y_true, best_pred)
    print("Confusion Matrix:\n", cm)

    if save_result:
        # fusion weight sweep
        plt.figure()
        plt.plot(weights, acc_list, marker='o')
        plt.xlabel("Fusion Weight (BiLSTM)")
        plt.ylabel("Accuracy")
        plt.title("Fusion Weight Sweep")
        plt.grid(True)
        plt.savefig(f"{FUSION_RESULT_DIR}/fusion_weight_sweep.png")
        plt.close()

        # confusion matrix
        plt.figure(figsize=(6, 5))
        classes = np.unique(y_true)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Fusion Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{FUSION_RESULT_DIR}/fusion_confusion_matrix.png")
        plt.close()

        # model accuracy compare
        acc_fusion = best_acc
        plt.figure(figsize=(6, 4))
        plt.bar(["BiLSTM", "TrackNet", "Fusion"], [acc_seq, acc_track, acc_fusion],
                color=["blue", "orange", "green"])
        plt.ylim(0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        plt.savefig(f"{FUSION_RESULT_DIR}/model_accuracy_comparison.png")
        plt.close()

        # accuracy in each class
        print("Per-Class Accuracy:")
        categories = np.unique(y_true)
        class_accs = []
        for cls in categories:
            cls_mask = y_true == cls
            cls_acc = accuracy_score(np.array(y_true)[cls_mask], np.array(best_pred)[cls_mask])
            class_accs.append(cls_acc)
            print(f"  Class {cls}: {cls_acc:.4f}")

        plt.figure(figsize=(6, 4))
        plt.bar([str(c) for c in categories], class_accs, color='purple')
        plt.ylim(0, 1.0)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy (Fusion)")
        plt.savefig(f"{FUSION_RESULT_DIR}/per_class_accuracy.png")
        plt.close()

        # save predict results
        np.savez(f"{FUSION_RESULT_DIR}/fusion_results.npz", y_true=y_true, y_pred=best_pred)
    print("[DEBUG] acc_seq =", acc_seq)
    print("[DEBUG] acc_track =", acc_track)
    print("[DEBUG] acc_fusion =", best_acc)

# === Run All ===
if __name__ == "__main__":
    X_seq, y_seq, track_ids_seq, X_track, y_track, track_ids_track = load_data()
    track_label_map = build_track_id_to_label_map(track_ids_track, y_track)
    pred_seq, pred_track, y_true = predict_with_models(
        X_seq, X_track, track_ids_seq, track_ids_track, track_label_map)


    train_and_evaluate_fusion(pred_seq, pred_track, y_true)
