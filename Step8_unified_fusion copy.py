import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd
from Config import EPOCHS, MODEL_DIR, BATCH_SIZE, TRACK_OUTPUT_SIZE
from Step3A_spotmodel import Attention
import random
from collections import defaultdict

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, "unified_model_best.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetDataset(Dataset):
    def __init__(self, seq_path, track_path, annotations_path, case_identifier, transform=None):
        X_seq, X_track, y_matched, prefix_tid = select_specific_cases(seq_path, track_path, annotations_path, case_identifier)

        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_track = torch.tensor(X_track, dtype=torch.float32)
        self.prefix_tid = prefix_tid
        self.transform = transform  

    def __len__(self):
        # Total number of samples
        return len(self.prefix_tid)

    def __getitem__(self, idx):
        seq = self.X_seq[idx]
        track = self.X_track[idx]
        prefix_tid = self.prefix_tid[idx]

        # Apply optional transform to features
        if self.transform:
            seq, track = self.transform((seq, track))

        return seq, track, prefix_tid 

def select_specific_cases(seq_path, track_path, annotations_path, case_identifier):
    specfic_cases = []
    annotations_df = pd.read_excel(annotations_path)
    specfic_cases = annotations_df.loc[
        annotations_df["Train or Test"] == case_identifier, "Case"
    ].tolist()

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

    X_seq_matched, X_track_matched, y_matched, prefix_tid = [], [], [], []
    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            prefix_split = "_".join(tid[0].split("_")[:2])
            if prefix_split in specfic_cases:
                idx = track_id_to_index[key]
                X_seq_matched.append(X_seq[i])
                X_track_matched.append(X_track[idx])
                y_matched.append(y_seq[i])
                prefix_tid.append(tid[0]+str(tid[1]))

    return X_seq_matched, X_track_matched, y_matched, prefix_tid


def train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path):
    annotations_df = pd.read_excel(test_train_split_annotation_path)
    train_cases = annotations_df.loc[annotations_df["Train or Test"] == 0, "Case"].tolist()
    test_cases  = annotations_df.loc[annotations_df["Train or Test"] == 1, "Case"].tolist()


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


    test_label_dist_dict = defaultdict(int)
    train_label_dist_dict = defaultdict(int)

    test_label_value_dist_dict = defaultdict(int)
    train_label_value_dist_dict = defaultdict(int)

    X_seq_matched_train, X_track_matched_train, y_matched_train, prefix_tid_train = [], [], [], []
    X_seq_matched_test, X_track_matched_test, y_matched_test, prefix_tid_test = [], [], [], []

    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            idx = track_id_to_index[key]
            case_name = "_".join(tid[0].split("_")[:2])

            if case_name in test_cases:
                X_seq_matched_test.append(X_seq[i])
                X_track_matched_test.append(X_track[idx])
                y_matched_test.append(y_seq[i])
                test_label_dist_dict[case_name] += 1
                test_label_value_dist_dict[y_seq[i]] += 1
                # prefix_tid_test.append(tid[0]+str(tid[1]))
            elif case_name in train_cases or not train_cases:
                X_seq_matched_train.append(X_seq[i])
                X_track_matched_train.append(X_track[idx])
                y_matched_train.append(y_seq[i])
                train_label_dist_dict[case_name] +=1
                train_label_value_dist_dict[y_seq[i]] += 1
                # prefix_tid_train.append(tid[0]+str(tid[1]))
            
 

    print(f"[DEBUG] Matched train pairs: {len(X_seq_matched_train)}")
    print(train_label_dist_dict)

    print(f"[DEBUG] Matched test pairs: {len(X_seq_matched_test)}")
    print(test_label_dist_dict)
    
    return (np.array(X_seq_matched_train), np.array(X_seq_matched_test), 
            np.array(X_track_matched_train), np.array(X_track_matched_test), 
            np.array(y_matched_train), np.array(y_matched_test))


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

    X_seq_matched, X_track_matched, y_matched, prefix_tid = [], [], [], []
    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            idx = track_id_to_index[key]
            X_seq_matched.append(X_seq[i])
            X_track_matched.append(X_track[idx])
            y_matched.append(y_seq[i])
            prefix_tid.append(tid[0]+str(tid[1]))

    print(f"[DEBUG] Matched pairs: {len(X_seq_matched)}")
    print(np.array(X_track_matched).shape)
    return np.array(X_seq_matched), np.array(X_track_matched), np.array(y_matched), prefix_tid


def count_params(module):
    return sum(p.numel() for p in module.parameters())

class UnifiedFusionModel(nn.Module):
    def __init__(self, seq_input_size, track_input_size, hidden_size=128, fusion_size=128, dropout=0.5):
        super().__init__()
        self.track_input_size = track_input_size

        self.lstm = nn.LSTM(input_size=seq_input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_size*2, dropout)
        self.norm = nn.LayerNorm(hidden_size*2)

        print("Total Sequence parameters:", count_params(self.lstm) + count_params(self.attn) + count_params(self.norm))

        if track_input_size > 0:
            # self.track_fc = nn.Sequential(
            #     nn.Linear(track_input_size, 266),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(266, 266),
            #     nn.ReLU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(266, TRACK_OUTPUT_SIZE),
            #     nn.LayerNorm(TRACK_OUTPUT_SIZE)
            # )

            self.track_fc = nn.Sequential(
                nn.Linear(track_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, TRACK_OUTPUT_SIZE),
                #nn.LayerNorm(TRACK_OUTPUT_SIZE)
            )

            # self.track_fc = nn.Sequential(
            #     nn.Linear(track_input_size, 266),
            #     nn.ReLU(),
            #     nn.Linear(266, 266),
            #     nn.ReLU(),
            #     nn.Linear(266, TRACK_OUTPUT_SIZE),
            #     nn.LayerNorm(TRACK_OUTPUT_SIZE),
            #     #nn.Dropout(0.3),
            # )

            print("Total Track parameters:", count_params(self.track_fc))

            self.use_track = True
        else:
            self.use_track = False

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + TRACK_OUTPUT_SIZE, fusion_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_size, 3)
        )

        # self.fusion_fc = nn.Sequential(
        #     nn.Linear(hidden_size * 2 + TRACK_OUTPUT_SIZE, FUSION_SIZE),
        #     nn.ReLU(),
        #     nn.LayerNorm(FUSION_SIZE),
        #     nn.Linear(FUSION_SIZE, 3)
        # )


    def forward(self, x_seq, x_track, lstm_weight=0.5):
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = self.norm(lstm_out)
        lstm_feat, attn_weights = self.attn(lstm_out)
        #lstm_feat = self.norm(lstm_feat)
        lstm_feat = F.layer_norm(lstm_feat, lstm_feat.shape[1:])
        lstm_feat = lstm_feat * lstm_weight
        

        if self.use_track:
            track_feat = self.track_fc(x_track)
            track_feat = F.layer_norm(track_feat, track_feat.shape[1:])
            track_feat = track_feat * (1 - lstm_weight)
            fused = torch.cat([lstm_feat, track_feat], dim=1)
        else:
            fused = lstm_feat
        return self.fusion_fc(fused)
    

def run_inference(model, X_seq, X_track, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_seq.to(device), X_track.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs


def compute_metrics(y_true, preds, probs):
    acc = np.mean(preds == y_true)
    f1 = f1_score(y_true, preds, average="macro")

    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    try:
        auc_value = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovo")
    except:
        auc_value = -1
    return acc, f1, auc_value, y_true_bin


def plot_roc(y_true_bin, probs, result_path, n_classes=3):
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(result_path,"roc_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, preds, classes, result_path):
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    plt.close()
    return cm


def fusion_weight_analysis(model, test_loader, device, result_path):
    def evaluate_model(model, dataloader, alpha, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_seq, batch_track, batch_y in dataloader:
                batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
                outputs = model(batch_seq, batch_track, lstm_weight=alpha)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        return correct / total
    alphas = np.linspace(0, 1, 21)
    accuracies = []
    for a in alphas:
        acc = evaluate_model(model, test_loader, a, device=device)
        accuracies.append(acc)
        print(f"Alpha={a:.2f}, Accuracy={acc:.4f}")
    plt.plot(alphas, accuracies, marker='o')
    plt.xlabel("LSTM Weighting")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sequence/Track Mixing Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "Fusion Weights.png"))
    plt.close()


def compute_case_proportions(model, dataset, device, batch_size, result_path):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    prefix_dict = {}
    for batch_seq, batch_track, batch_prefix_tid in loader:
        batch_seq, batch_track = batch_seq.to(device), batch_track.to(device)
        logits = model(batch_seq, batch_track)
        pred = logits.argmax(dim=1)
        for i in range(len(batch_prefix_tid)):
            case_id = "_".join(batch_prefix_tid[i].split('_')[:2])
            if case_id not in prefix_dict:
                prefix_dict[case_id] = [0, 0, 0]
            prefix_dict[case_id][pred[i]] += 1

    df = pd.DataFrame(prefix_dict).transpose()
    df.columns = ['Progressive', 'Stable', 'Responsive']
    df = df.div(df.sum(axis=1), axis=0).sort_index()

    ax = df.plot(kind='barh', stacked=True,
                 title='T cell Proportions by Case',
                 color=['#90BFF9', '#FFC080', '#FFA0A0'])
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    plt.savefig(os.path.join(result_path, "proportions_by_case.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return df

def correlate_with_size_change(df, annotations_path, result_path):
    size_df = pd.read_excel(annotations_path)
    size_dict = size_df.set_index("Case")["Size Change"].to_dict()

    x, y = [], []
    for case_name in df.index:
        y.append(df.loc[case_name, "Combined Score"])
        x.append(size_dict[case_name])

    x, y = np.array(x), np.array(y)
    df["Size Change"] = x
    df.to_csv(os.path.join(result_path, "proportions.csv"), index=True)

    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
    r2 = r2_score(y, y_pred)

    plt.scatter(x, y)
    plt.plot(x, y_pred, color="red", label=f"Best fit (R²={r2:.2f})")
    plt.xlabel("Change in PDO size")
    plt.ylabel("Score")
    plt.title("Score by Change in PDO size")
    plt.legend()
    plt.savefig(os.path.join(result_path, "Score by Change in PDO size.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return r2

def plot_loss_curve(train_losses, val_losses, results_path):
    print("[STEP 2] Drawing Loss Graph...")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{results_path}/loss_curve.png")
    plt.close()
    print("[STEP 2] Finished Drawing Loss Graph...")

def plot_val_accuracy(val_accs, results_path):
    print("[STEP 2] Drawing Validation Accuracy Graph...")
    plt.plot(np.array(val_accs) * 100)
    plt.title("Validation Accuracy (%)")
    plt.savefig(f"{results_path}/val_accuracy.png")
    plt.close()
    print("[STEP 2] Finished Drawing Validation Accuracy Graph...")

def Train_UnifiedFusionModel(seq_path, track_path, model_save_path, result_path,
                             seq_input_size=9, track_input_size=12, hidden_size=128, fusion_size=128, dropout=0.5, test_prefix="no_prefix"):
    
    print("[STEP 1] Loading and aligning data...")
    test_train_split_annotation_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Data\Annotations.xlsx"
    X_seq_train, X_seq_test, X_track_train, X_track_test, y_train, y_test_original = train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path=test_train_split_annotation_path)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test_original)

    train_classes = set(le.classes_)
    test_classes = set(y_test_original)
    unknown_labels = test_classes - train_classes
    if (unknown_labels):
        print("Unknown labels label found in test set but not in train:", unknown_labels)

    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)
    X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    X_track_test = torch.tensor(X_track_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_seq_train, X_track_train, y_train_tensor)
    test_dataset = TensorDataset(X_seq_test, X_track_test, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=0, pin_memory=False)

    classes, counts = np.unique(y_train, return_counts=True)
    print("Class counts (via numpy):")
    for cls, count in zip(classes, counts):
        print(f"  Class {cls}: {count} samples")

    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=classes, 
        y=y_train
    )

    # Convert to torch tensor
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)  # move to GPU if needed
    print("Class Weights:", weights)

    model = UnifiedFusionModel(seq_input_size=seq_input_size, track_input_size=track_input_size,
                               hidden_size=hidden_size, fusion_size=fusion_size, dropout=dropout).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    #criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    print("[STEP 2] Training unified fusion model...")
    best_acc, early_stop = 0, 0
    entropy_total, entropy_count = 0.0, 0
    lowest_loss = 10000
    best_model = None
    train_accs, train_losses, val_losses, val_accs = [], [], [], []

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(EPOCHS):
        model.train()
        correct_train, total_train, train_loss = 0, 0, 0
        for batch_seq, batch_track, batch_y in train_loader:
            batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
            optimizer.zero_grad()
            # logits = model(batch_seq, batch_track)
            # loss = criterion(logits, batch_y)
            # loss.backward()
            # optimizer.step()
            with torch.amp.autocast("cuda"):
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct_train += (pred == batch_y).sum().item()
            total_train += batch_y.size(0)

        model.eval()
        correct_val, total_val, val_loss = 0, 0, 0
        with torch.no_grad():
            for batch_seq, batch_track, batch_y in test_loader:
                batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                # probs = F.softmax(logits, dim=1)
                # batch_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                # entropy_total += batch_entropy.sum().item()
                # entropy_count += batch_entropy.size(0)
                # avg_entropy = entropy_total / entropy_count
                # print(f"[VAL] Epoch {epoch+1} - Avg Prediction Entropy: {avg_entropy:.4f}")

                pred = logits.argmax(dim=1)
                correct_val += (pred == batch_y).sum().item()
                total_val += batch_y.size(0)



        train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        val_loss /= len(test_loader)
        val_acc = correct_val / total_val
        
        scheduler.step(val_loss)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1} | Loss = {train_loss:.4f} | Val Loss={val_loss:.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
            print(scheduler.get_last_lr())

        if val_loss < lowest_loss:
            lowest_loss, best_model = val_loss, model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 60:
                print("Early stopping triggered.")
                break

    # -------------------------------------------------
    # TRAINING LOSS/ACCURACY GRAPHS
    # -------------------------------------------------

    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

    train_results_path = os.path.join(result_path, f"train results/hidden_{hidden_size}/fusion_{fusion_size}")
    os.makedirs(train_results_path, exist_ok=True)

    model.load_state_dict(best_model)
    
    lstm_weight_norm = sum(p.norm().item() for n, p in model.lstm.named_parameters() if 'weight' in n)
    track_weight_norm = sum(p.norm().item() for n, p in model.track_fc.named_parameters() if 'weight' in n)
    print("LSTM weight norm:", lstm_weight_norm)
    print("Track weight norm:", track_weight_norm)
    
    
    np.savez(f"{train_results_path}/training_logs_unified.npz", 
             train_losses=train_losses, train_accs=train_accs,
             val_losses=val_losses, val_accuracies=val_accs)
    
    # Plot training curve
    plot_loss_curve(train_losses, val_losses, train_results_path)
    plot_val_accuracy(val_accs, train_results_path)
    

    print("[STEP 3] Evaluating...")

    # -------------------------------------------------
    # BEST TRAINING GRAPHS
    # -------------------------------------------------


    preds, probs = run_inference(model, X_seq_train, X_track_train, device)
    best_train_acc, f1, auc_value, y_train_bin = compute_metrics(y_train, preds, probs)

    print(f"[RESULT] Accuracy: {best_train_acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")
    print(classification_report(y_train, preds, target_names=[str(cls) for cls in le.classes_]))

    plot_roc(y_train_bin, probs, train_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_train, preds, le.classes_, train_results_path)

    fusion_weight_analysis(model, train_loader, device, train_results_path)

    df = compute_case_proportions(model, SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 0),
                                device, BATCH_SIZE, train_results_path)
    df["Combined Score"] = (df["Progressive"]*0 + df["Stable"]*0.5 + df["Responsive"]*1.0)

    r2 = correlate_with_size_change(df, test_train_split_annotation_path, train_results_path)
    print(f"[RESULT] R² correlation with size change = {r2:.3f}")


    # -------------------------------------------------
    # BEST VALIDATION/TEST GRAPHS
    # -------------------------------------------------
    test_results_path = os.path.join(result_path, f"val results/hidden_{hidden_size}/fusion_{fusion_size}")
    os.makedirs(test_results_path, exist_ok=True)

    preds, probs = run_inference(model, X_seq_test, X_track_test, device)
    best_val_acc, f1, auc_value, y_test_bin = compute_metrics(y_test, preds, probs)

    print(f"[RESULT] Accuracy: {best_val_acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")
    print(classification_report(y_test, preds, target_names=[str(cls) for cls in le.classes_]))

    plot_roc(y_test_bin, probs, test_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_test, preds, le.classes_, test_results_path)

    fusion_weight_analysis(model, test_loader, device, test_results_path)

    df = compute_case_proportions(model, SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 1),
                                device, BATCH_SIZE, test_results_path)
    df["Combined Score"] = (df["Progressive"]*0 + df["Stable"]*0.5 + df["Responsive"]*1.0)

    r2 = correlate_with_size_change(df, test_train_split_annotation_path, test_results_path)
    print(f"[RESULT] R² correlation with size change = {r2:.3f}")

    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

    return {
        "f1_score": f1,
        "auc": auc_value,
        "confusion_matrix": cm.tolist(),
        "train_losses": train_losses,
        "train_accuracy": train_accs,
        "best_train_acc": best_train_acc,
        "val_losses": val_losses,
        "val_accuracy": val_accs,
        "best_val_acc": best_val_acc,
        "r2": r2
    }



def Test_UnifiedFusionModel(seq_path, track_path, model_path, results_dir="test", seq_input_size=9, track_input_size=12, hidden_size=128, fusion_size=128, dropout=0.5,):
    print("[TEST] Loading external test dataset...")

    test_train_split_annotation_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Data\Annotations.xlsx"
    X_seq_train, X_seq_test, X_track_train, X_track_test, y_train, y_test_original = train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path=test_train_split_annotation_path)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test_original)

    train_classes = set(le.classes_)
    test_classes = set(y_test_original)
    unknown_labels = test_classes - train_classes
    if (unknown_labels):
        print("Unknown labels label found in test set but not in train:", unknown_labels)

    X_seq_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
    X_track_tensor = torch.tensor(X_track_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_seq_tensor, X_track_tensor, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UnifiedFusionModel(seq_input_size=seq_input_size, track_input_size=track_input_size,
                               hidden_size=hidden_size, fusion_size=fusion_size, dropout=dropout).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # -------------------------------------------------
    # Evaluation Graphs
    # -------------------------------------------------
    test_results_path = os.path.join(results_dir, f"test results/hidden_{hidden_size}/fusion_{fusion_size}")

    print("[STEP 3] Evaluating...")
    preds, probs = run_inference(model, X_seq_tensor, X_track_tensor, device)
    acc, f1, auc_value, y_test_bin = compute_metrics(y_test, preds, probs)

    print(f"[RESULT] Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")
    print(classification_report(y_test, preds, target_names=[str(cls) for cls in le.classes_]))

    plot_roc(y_test_bin, probs, test_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_test, preds, le.classes_, test_results_path)

    fusion_weight_analysis(model, test_loader, device, test_results_path)

    df = compute_case_proportions(model, SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 1),
                                device, BATCH_SIZE, test_results_path)
    df["Combined Score"] = (df["Progressive"]*0 + df["Stable"]*0.5 + df["Responsive"]*1.0)

    r2 = correlate_with_size_change(df, test_train_split_annotation_path, test_results_path)
    print(f"[RESULT] R² correlation with size change = {r2:.3f}")

    return {
        "f1_score": f1,
        "auc": auc_value,
        "confusion_matrix": cm.tolist(),
        "acc": acc,
        "r2": r2
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