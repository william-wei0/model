import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd

# === model and variable import ===
from Config import *
from Step8_unified_fusion import UnifiedFusionModel, load_and_align_data
torch.backends.cudnn.enabled = False

total_seq = SEQ_LEN * FEATURE_LEN # e.g. 180 = 20 * 9

# === routes configuration ===
SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
MODEL_SAVE_PATH = f"{MODEL_DIR}/unified_fusion_model.pth"
os.makedirs(SEQ_RESULT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === load data and model  ===
print("[STEP 1] Loading model and data...")

model = UnifiedFusionModel().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

X_seq, X_track, y = load_and_align_data()
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_seq_train, X_seq_test, X_track_train, X_track_test, y_train, y_test = train_test_split(
    X_seq, X_track, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)
X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
X_track_test = torch.tensor(X_track_test, dtype=torch.float32)


# combine (X_seq, X_track) to one tensor
X_seq_flat = X_seq_test.reshape(X_seq_test.shape[0], -1)  # (N, 20*9)
X_test_concat = torch.cat([X_seq_flat, X_track_test], dim=1).to(device)

X_train_seq_flat = X_seq_train.reshape(X_seq_train.shape[0], -1)
X_train_concat = torch.cat([X_train_seq_flat, X_track_train], dim=1).to(device)

# define wrapped model
class WrappedUnifiedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_concat):
        # x_concat: (100, 192) 192 = 9*T + 12
        batch_size = x_concat.shape[0]
        x_seq_flat = x_concat[:, :total_seq]  # T=20, F=9
        x_track = x_concat[:, total_seq:]
        
        x_seq = x_seq_flat.view(batch_size, SEQ_LEN, 9)
        return self.model(x_seq, x_track)
    
# define shap
print("[STEP 2] SHAP analysis...")
wrapped_model = WrappedUnifiedModel(model).to(device)
explainer = shap.GradientExplainer(wrapped_model, X_train_concat[:100])
shap_values = explainer.shap_values(X_test_concat[:100], ranked_outputs=1)

print("[STEP 3] SHAP drawing...")
# open (100, total_seq+12, 1) 
# get shap value from package 
shap_values_combined = shap_values[0]

shap_value_seq = shap_values_combined[:, :total_seq]
shap_value_track = shap_values_combined[:, total_seq:]

shap_value_seq = shap_value_seq.reshape(100, SEQ_LEN, FEATURE_LEN)

shap_result_seq = shap_value_seq.mean(axis=(0,1))
shap_result_track = shap_value_track.mean(axis=(0,2))

feature_names = features + track_features

shap_result = np.concatenate((shap_result_seq, shap_result_track)) 


shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": shap_result
}).sort_values("Importance", ascending=False)



plt.figure(figsize=(12, 6))
sns.barplot(data=shap_df.head(30), x="Importance", y="Feature")
plt.title("Unified fusion SHAP Feature Importances")
plt.tight_layout()
plt.savefig(f"{SEQ_RESULT_DIR}/unified_shap_bar.png")
print("SHAP feature importance saved.")
