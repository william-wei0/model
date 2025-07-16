import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for environments without GUI
import matplotlib.pyplot as plt
from Config import GENERATED_DIR, features, track_features

save_dir = os.path.join(GENERATED_DIR, "feature_distribution")
os.makedirs(save_dir, exist_ok=True)


def plot_feature_distribution_by_label(feature_name):
    """
    Plot the mean and standard deviation of a given feature grouped by label.

    Args:
        feature_name (str): The name of the feature to plot. Should be in either features or track_features.
    """

    # Check if the feature is valid and determine its data level
    valid_features = track_features + features
    is_track_level = feature_name in track_features
    if feature_name not in valid_features:
        raise ValueError(f"Invalid feature name: {feature_name}. Must be one of: {valid_features}")

    # Load unnormalized data
    csv_file = "unscaled_track_features.csv" if is_track_level else "unscaled_spot_features.csv"
    df = pd.read_csv(os.path.join(GENERATED_DIR, csv_file))

    if feature_name not in df.columns or 'LABEL' not in df.columns:
        raise ValueError(f"'{feature_name}' or 'LABEL' not found in dataframe.")

    # Compute mean and standard deviation for each label
    stats = df.groupby("LABEL")[feature_name].agg(['mean', 'std', 'count']).reset_index()
    stats['sem'] = stats['std'] / stats['count']**0.5
    
    # Create bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(stats["LABEL"], stats["mean"], yerr=stats["sem"],
            width=0.5, capsize=8, alpha=0.7, color='skyblue')
    plt.xlabel("Label")
    plt.ylabel(f"{feature_name} Mean Value")
    plt.title(f"{feature_name} Mean ± SEM by Label")
    plt.xticks(stats["LABEL"])

    # Annotate each bar with its value
    for i, row in stats.iterrows():
        plt.text(row["LABEL"], row["mean"] + row["sem"] * 0.05, f'{row["mean"]:.2f}', 
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save plot to file
    out_path = os.path.join(save_dir, f"feature_{feature_name}_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")

# Example usage
if __name__ == "__main__":
    for feature in features + track_features:
        plot_feature_distribution_by_label(feature)
    
