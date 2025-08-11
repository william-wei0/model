import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load your dataframe
df = pd.read_csv(r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\unscaled_spot_features.csv")  # Replace with actual file path
# Define the columns to average
columns_to_average = ["RADIUS", "AREA", "PERIMETER", "CIRCULARITY", 
                      "ELLIPSE_MAJOR", "ELLIPSE_MINOR", "ELLIPSE_ASPECTRATIO", 
                      "SOLIDITY", "SPEED", "MEAN_SQUARE_DISPLACEMENT"]

# Group by PREFIX and TRACK_ID
grouped = df.groupby(["PREFIX", "TRACK_ID"])

# Compute the average of each feature
averaged = grouped[columns_to_average].mean().reset_index()

# Add number of frames per track
averaged["FRAME_COUNT"] = grouped.size().values

# ------------------------------------
# Compute alpha (slope of log(MSD) vs log(time/frame))
# ------------------------------------
alpha_values = []

for (prefix, track_id), group in grouped:
    group_sorted = group.sort_values("FRAME")
    
    time = group_sorted["FRAME"].values
    msd = group_sorted["MEAN_SQUARE_DISPLACEMENT"].values
    
    # Remove non-positive values to avoid log errors
    valid = (time > 0) & (msd > 0)
    if valid.sum() < 2:
        alpha = np.nan
    else:
        log_t = np.log(time[valid])
        log_msd = np.log(msd[valid])
        slope, _, _, _, _ = linregress(log_t, log_msd)
        alpha = slope

        if alpha < 0 or alpha > 5:
            alpha = np.nan

    # Extract label (assuming it’s the same for all frames in the group)
    label = group_sorted["LABEL"].iloc[0]

    alpha_values.append({
        "PREFIX": prefix,
        "TRACK_ID": track_id,
        "ALPHA": alpha,
        "LABEL": label
    })

# Create DataFrame for alpha + label
alpha_df = pd.DataFrame(alpha_values)

# Merge with averaged features
final_df = pd.merge(averaged, alpha_df, on=["PREFIX", "TRACK_ID"])


final_df.to_csv(r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\averaged_features_by_track_msd.csv", index=False)