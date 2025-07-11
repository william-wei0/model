import os
import pandas as pd
import matplotlib.pyplot as plt

from Config import DATA_DIR, GENERATED_DIR, MODEL_DIR, RESULTS_DIR, SEQ_LEN
from Step1_data import Create_Dataset
from Step8_unified_fusion import Train_UnifiedFusionModel
from Step9_SHAP_Unified import SHAP_UnifiedFusionModel

# === 定义特征配置 ===
all_features = [  # time-based
    'RADIUS', 'AREA', 'PERIMETER', 'CIRCULARITY',
    'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_ASPECTRATIO', 'SOLIDITY', 'SPEED'
]

all_track_features = [  # track-based
    "TRACK_DURATION", "TRACK_DISPLACEMENT", "TRACK_MEAN_SPEED",
    "TRACK_MAX_SPEED", "TRACK_MIN_SPEED", "TRACK_STD_SPEED",
    "TOTAL_DISTANCE_TRAVELED", "MAX_DISTANCE_TRAVELED", "CONFINEMENT_RATIO",
    "MEAN_STRAIGHT_LINE_SPEED", "LINEARITY_OF_FORWARD_PROGRESSION",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]

ablation_configs = {
    "baseline": {
        "features": all_features,
        "track_features": all_track_features
    },
    "remove_negative": {
        "features": all_features,
        "track_features": [
            f for f in all_track_features if f not in [
                "TRACK_MEAN_SPEED", "TRACK_STD_SPEED", "MEAN_STRAIGHT_LINE_SPEED"
            ]
        ]
    },
    "top6_shap": {
        "features": ["RADIUS", "SOLIDITY", "ELLIPSE_MINOR"],
        "track_features": ["TRACK_DURATION", "TRACK_STD_SPEED", "TRACK_MEAN_SPEED"]
    }
}

# === 结果记录列表 ===
results_summary = []

# === 开始实验 ===
for name, cfg in ablation_configs.items():
    print(f"\n===== Running Ablation: {name} =====")

    # 路径设置
    prefix = f"ablation_{name}"
    output_prefix = prefix
    model_path = os.path.join(MODEL_DIR, f"{prefix}.pth")
    result_path = os.path.join(RESULTS_DIR, prefix)
    os.makedirs(result_path, exist_ok=True)

    # dataset route
    seq_path = os.path.join(GENERATED_DIR, f"{output_prefix}_{SEQ_LEN}.npz")
    track_path = os.path.join(GENERATED_DIR, f"{output_prefix}track_dataset.npz")

    # Step1: 创建数据集（含多 SEQ_LEN）
    Create_Dataset(
        DATA_DIR,
        GENERATED_DIR,
        cfg["features"],
        cfg["track_features"],
        output_prefix=output_prefix,
        default_seq_len=[SEQ_LEN]
    )

    # Step2: 训练模型（返回准确率等）
    seq_input_size = len(cfg["features"])
    track_input_size = len(cfg["track_features"])

    metrics = Train_UnifiedFusionModel(
        seq_path=seq_path,
        track_path=track_path,
        model_save_path=model_path,
        result_path=result_path,
        seq_input_size=seq_input_size,
        track_input_size=track_input_size,
        hidden_size=64,
        dropout=0.0
    )

    # Step3: SHAP 分析
    SHAP_UnifiedFusionModel(
        seq_length=SEQ_LEN,
        features=cfg["features"],
        track_features=cfg["track_features"],
        model_save_path=model_path,
        result_path=result_path,
        seq_path=seq_path,
        track_path=track_path
    )

    # Step4: 记录结果
    results_summary.append({
        "config_name": name,
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1_score"],
        "auc": metrics["auc"]
    })

# === 输出对比表格 ===
df = pd.DataFrame(results_summary)
df = df.sort_values(by="accuracy", ascending=False)
print("\n=== Summary of Ablation Results ===")
print(df)

# === 保存CSV ===
summary_path = os.path.join(RESULTS_DIR, f"ablation_summary_{SEQ_LEN}.csv")
df.to_csv(summary_path, index=False)
print(f"Saved summary to {summary_path}")

# === 可选：绘图可视化 ===
plt.figure(figsize=(10, 5))
df.plot(x='config_name', y=['accuracy', 'f1_score', 'auc'], kind='bar')
plt.title(f"Ablation Performance Comparison (SeqLen={SEQ_LEN})")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"ablation_barplot_{SEQ_LEN}.png"))
plt.close()
print("Saved performance bar plot.")
