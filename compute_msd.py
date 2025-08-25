import pandas as pd
import numpy as np
from scipy.stats import linregress

# Load your dataframe



def swap_columns(df, col1, col2):
    cols = list(df.columns)
    i, j = cols.index(col1), cols.index(col2)
    cols[i], cols[j] = cols[j], cols[i]
    df = df[cols]
    return df

def calc_a():
    df_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\unscaled_spot_features_CAF.csv"
    output_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\averaged_features_by_track_msd.csv"

    merge_track = True
    track_df_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\unscaled_track_features_CAF.csv"
    orig_spots_df = pd.read_csv(df_path)  # Replace with actual file path

    # ------------------------------------
    # Compute alpha (slope of log(MSD) vs log(time/frame))
    # ------------------------------------
    alpha_values = []
    for (prefix, track_id), group in orig_spots_df.groupby(["PREFIX", "TRACK_ID"]):
        # only use lag >= 1 and positive MSD
        valid = (group["FRAME"] > 0) & (group["MEAN_SQUARE_DISPLACEMENT"] > 0)
        g = group.loc[valid]
        g["FRAME"] = g["FRAME"] - g["FRAME"].iloc[0] + 1

        if len(g) < 2:  # need at least 2 points to fit
            alpha = np.nan
        else:
            slope, _, _, _, _ = linregress(
                np.log(g["FRAME"]),
                np.log(g["MEAN_SQUARE_DISPLACEMENT"])
            )
            alpha = slope

        alpha_values.append({
            "PREFIX": prefix,
            "TRACK_ID": track_id,
            "ALPHA": alpha,
            "LABEL": group["LABEL"].iloc[0]
        })

    alpha_df = pd.DataFrame(alpha_values)


    print(f"Calulated averages succesfully to {output_path}")
    alpha_df.to_csv(output_path, index=False)

def calc_b():
    df_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\msd.csv"
    output_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\averaged_features_by_track_msd.csv"

    merge_track = True
    track_df_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\unscaled_track_features_CAF.csv"
    orig_spots_df = pd.read_csv(df_path)  # Replace with actual file path

    # ------------------------------------
    # Compute alpha (slope of log(MSD) vs log(time/frame))
    # ------------------------------------
    alpha_values = []
    for (prefix, track_id), group in orig_spots_df.groupby(["PREFIX", "TRACK_ID"]):
        # only use lag >= 1 and positive MSD
        valid = (group["LAG"] > 0) & (group["MSD"] > 0)
        g = group.loc[valid]
        print(g)

        if len(g) < 2:  # need at least 2 points to fit
            alpha = np.nan
        else:
            slope, _, _, _, _ = linregress(
                np.log(g["LAG"]),
                np.log(g["MSD"])
            )
            alpha = slope

        alpha_values.append({
            "PREFIX": prefix,
            "TRACK_ID": track_id,
            "ALPHA": alpha,
            "LABEL": group["LABEL"].iloc[0]
        })

    alpha_df = pd.DataFrame(alpha_values)


    print(f"Calulated averages succesfully to {output_path}")
    alpha_df.to_csv(output_path, index=False)

calc_b()