import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg') 

from Step2_trackmodel import TrackNet
from Config import *

# === Config ===
MODEL_PATH = f"{MODEL_DIR}/track_model.pth"
DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"

# === Step 1: Load dataset ===
print("[STEP 1] Loading data...")
data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]
print(f"[DEBUG] Loaded X shape: {X.shape} | y shape: {y.shape}")

X = torch.tensor(X, dtype=torch.float32)
from sklearn.preprocessing import LabelEncoder
y_encoded = LabelEncoder().fit_transform(y)
y_encoded = torch.tensor(y_encoded, dtype=torch.long)

# === Step 2: Split dataset ===
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# === Step 3: Define and load model ===
class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        return self.base(x)

model = TrackNet(input_dim=X.shape[1], num_classes=len(torch.unique(y_encoded)))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
wrapped_model = WrappedModel(model)

# === Step 4: Select background and test samples ===
background = X_train[:100]
test_samples = X_test[:100]

print("[INFO] Running SHAP analysis...")
explainer = shap.GradientExplainer(wrapped_model, background)
shap_values = explainer.shap_values(test_samples, ranked_outputs=3)

# === Step 5: Visualize every class SHAP feature importance ===
shap_tensor = shap_values[0]  # shape: (N, F, C)
shap_tensor = shap_tensor.cpu().numpy() if isinstance(shap_tensor, torch.Tensor) else shap_tensor

N, F, C = shap_tensor.shape
print(f"[DEBUG] SHAP value shape: {shap_tensor.shape}")

if F != len(track_features):
    print("[WARNING] Feature number mismatch, using default names.")
    features = [f"Feature {i}" for i in range(F)]
else:
    features = track_features

all_class_importances = []

for class_id in range(C):
    shap_class = shap_tensor[:, :, class_id]  # shape: (N, F)
    mean_shap = np.abs(shap_class).mean(axis=0)  # shape: (F,)
    all_class_importances.append(mean_shap)

    df = pd.DataFrame({
        "Feature": features,
        "Importance": mean_shap
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Importance", y="Feature")
    plt.title(f"Mean SHAP Feature Importance - Class {class_id}")
    plt.tight_layout()
    out_path = os.path.join(TRACK_RESULT_DIR, f"shap_bar_class{class_id}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved SHAP bar plot: {out_path}")

# === Mean importance plot ===
mean_importance_all_classes = np.mean(all_class_importances, axis=0)

df_mean = pd.DataFrame({
    "Feature": features,
    "Importance": mean_importance_all_classes
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_mean, x="Importance", y="Feature")
plt.title("Mean SHAP Feature Importance - Averaged Across Classes")
plt.tight_layout()
avg_path = os.path.join(TRACK_RESULT_DIR, "shap_bar_overall.png")
plt.savefig(avg_path)
plt.close()
print(f"[INFO] Saved overall SHAP importance plot: {avg_path}")