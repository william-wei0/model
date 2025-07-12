import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Config import DATA_DIR, GENERATED_DIR, features, track_features


# === Step 1: Load Annotations ===
def load_annotations(path, is_second_batch=False):
    if is_second_batch:
        df = pd.read_excel(path, sheet_name=0)
        id_series = df["Meso IL18 CAR T cells"].astype(str).str.strip()
        label_series = df["Labels"].astype(float)
    else:
        df = pd.read_excel(path, sheet_name="Summary")
        id_series = df.iloc[:, 1].astype(str).str.strip()
        label_series = df.iloc[:, 2].astype(float)

    mapping = dict(zip(id_series, label_series))
    print(f"Loaded annotation from {path}")
    print("Total entries:", len(mapping))
    return mapping


# === Step 2: Load Track/Spot Files ===
def load_tracks_and_spots(folder, cart_labels, second_labels):
    all_spots = []
    all_tracks = []

    for file_name in os.listdir(folder):
        if not file_name.endswith("_tracks.csv"):
            continue

        prefix = file_name.replace("_tracks.csv", "")
        spot_file = file_name.replace("_tracks.csv", "_spots.csv")

        track_path = os.path.join(folder, file_name)
        spot_path = os.path.join(folder, spot_file)

        if prefix.startswith("2nd_"):
            prefix_base = prefix.replace("2nd_", "").split("_")[0]
            label_dict = second_labels
        else:
            prefix_base = prefix.split("_")[0]
            label_dict = cart_labels

        if prefix_base not in label_dict:
            print(f"{prefix_base} cannot be found in annotation")
            continue

        try:
            df_raw_track = pd.read_csv(track_path, encoding='latin1',
                                       header=None)
            names_track = df_raw_track.iloc[0].tolist()
            df_track = pd.read_csv(track_path, encoding='latin1',
                                   skiprows=4, names=names_track)
            df_track['PREFIX'] = prefix

            df_raw_spot = pd.read_csv(spot_path, encoding='latin1',
                                      header=None)
            names_spot = df_raw_spot.iloc[0].tolist()
            df_spot = pd.read_csv(spot_path, encoding='latin1',
                                  skiprows=4, names=names_spot)
            df_spot['PREFIX'] = prefix
            df_spot['LABEL'] = label_dict[prefix_base]

            all_tracks.append(df_track)
            all_spots.append(df_spot)

        except Exception as e:
            print(f"Failed to load {prefix}: {e}")
            continue

    spots_df = pd.concat(all_spots, ignore_index=True)
    tracks_df = pd.concat(all_tracks, ignore_index=True)
    return spots_df, tracks_df


# === Step 3: Filter Valid Trajectories ===
def filter_valid_trajectories(spots_df, tracks_df, min_frames=10):
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames
                          ][["PREFIX", "TRACK_ID"]]
    spots_df_filtered = spots_df.merge(valid_ids, 
                                       on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_df_filtered = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]
    return spots_df_filtered, tracks_df_filtered


# === Step 4: Compute Features ===
def compute_features(spots_df):
    spots_df = spots_df.sort_values(by=["PREFIX", "TRACK_ID", "FRAME"])
    spots_df["VELOCITY_X"] = spots_df.groupby(
        ["PREFIX", "TRACK_ID"])["POSITION_X"].diff().fillna(0)
    spots_df["VELOCITY_Y"] = spots_df.groupby(
        ["PREFIX", "TRACK_ID"])["POSITION_Y"].diff().fillna(0)
    
    spots_df["SPEED"] = np.sqrt(spots_df["VELOCITY_X"]**2
                                + spots_df["VELOCITY_Y"]**2)
    spots_df["DIRECTION"] = np.arctan2(spots_df["VELOCITY_Y"],
                                       spots_df["VELOCITY_X"]) / np.pi

    drop_cols = [col for col in spots_df.columns 
                 if "INTENSITY" in col or col in ["POSITION_X", "POSITION_Y"]]
    spots_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    for f in features:
        spots_df[f] = pd.to_numeric(spots_df[f], errors='coerce')
    spots_df = spots_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return spots_df

    

# === Step 5: Align and Save Sequences ===
def align_and_save_dataset(spots_df, features, seq_len=20,
                           output_prefix="trajectory_dataset"):
    X_list, y_list, track_id_list = [], [], []
    rows = []

    for prefix, group_df in spots_df.groupby("PREFIX"):
        # features_except_dir = [f for f in features if f != "DIRECTION"]
        scaler = StandardScaler()
        # scaler.fit(group_df[features_except_dir])
        scaler.fit(group_df[features])
        for (p, tid), traj in group_df.groupby(["PREFIX", "TRACK_ID"]):
            feat = traj[features].values
            if len(feat) >= seq_len:
                feat = feat[:seq_len]
            else:
                pad = np.zeros((seq_len - len(feat), len(features)))
                feat = np.vstack([feat, pad])

            feat_scaled = feat.copy()
            df_temp = pd.DataFrame(feat, columns=features)
            feat_scaled = scaler.transform(df_temp)

            X_list.append(feat_scaled)
            y_list.append(traj["LABEL"].iloc[0])
            track_id_list.append((p, tid))

            for t in range(seq_len):
                row = [f"{p}_{tid}", t] + list(feat_scaled[t])
                rows.append(row)

    X = np.array(X_list)
    y = np.array(y_list)
    track_ids = np.array(track_id_list, dtype=object)
    np.savez(f"{GENERATED_DIR}/{output_prefix}_{seq_len}.npz",
             X=X, y=y, track_ids=track_ids)

    df_out = pd.DataFrame(rows, columns=["SampleID", "Frame"] + features)
    df_out.to_csv(f"{GENERATED_DIR}/{output_prefix}_{seq_len}.csv", index=False)

    print(
        f"Saved: {GENERATED_DIR}/{output_prefix}_{seq_len}.npz & .csv "
        f"| Shape: {X.shape}"
    )


# === Step 6: Save Track-Level Dataset ===
def build_track_level_dataset(tracks_df, cart_labels, second_labels,
                              output_prefix1="", 
                              track_features = track_features):
    if len(track_features) == 0:
        print("[INFO] No track features available.")
        return
    def match_label(prefix):
        if prefix.startswith("2nd_"):
            prefix_base = prefix.replace("2nd_", "").split("_")[0]
            return second_labels.get(prefix_base, np.nan)
        else:
            prefix_base = prefix.split("_")[0]
            return cart_labels.get(prefix_base, np.nan)

    tracks_df["LABEL"] = tracks_df["PREFIX"].apply(match_label)
    df = tracks_df.dropna(subset=track_features +
                          ["LABEL", "PREFIX", "TRACK_ID"]).copy()

    records = []
    for prefix, group in df.groupby("PREFIX"):
        scaler = StandardScaler()
        group_feat = group[track_features].values
        group_scaled = scaler.fit_transform(group_feat)

        for i, row in enumerate(group.itertuples()):
            record = {
                "PREFIX": prefix,
                "TRACK_ID": row.TRACK_ID,
                "LABEL": row.LABEL
            }
            for j, f in enumerate(track_features):
                record[f] = group_scaled[i][j]
            records.append(record)

    df_final = pd.DataFrame(records)
    df_final.to_csv(f"{GENERATED_DIR}/{output_prefix1}track_dataset.csv",
                    index=False)
    
    np.savez(f"{GENERATED_DIR}/{output_prefix1}track_dataset.npz", 
            X=df_final[track_features].values, 
            y=df_final["LABEL"].values,
            track_ids=df_final[["PREFIX", "TRACK_ID"]].values)
    print(f"Saved: {GENERATED_DIR}/{output_prefix1}track_dataset.csv & .npz")

    

if __name__ == "__main__":

    cart_labels = load_annotations(f"{DATA_DIR}/CART annotations.xlsx",
                                   is_second_batch=False)
    second_labels = load_annotations(f"{DATA_DIR}/2nd batch annotations.xlsx",
                                     is_second_batch=True)
    
    spots_df, tracks_df = load_tracks_and_spots(
        folder=f"{DATA_DIR}/TRACK",
        cart_labels=cart_labels,
        second_labels=second_labels
    )
    
    spots_df, tracks_df = filter_valid_trajectories(spots_df, tracks_df)
    spots_df = compute_features(spots_df)
    
    for seq_len_iter in [20,100,360]:
        align_and_save_dataset(spots_df,
                               features, seq_len=seq_len_iter,
                               output_prefix="trajectory_dataset")
    
    build_track_level_dataset(tracks_df, cart_labels, second_labels)    
    
    print("Dataset creation completed.")