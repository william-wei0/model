import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Config import DATA_DIR, GENERATED_DIR, features, track_features,SEQ_LEN
# def Create_Dataset(DATA_DIR, GENERATED_DIR, features, track_features,output_prefix,default_seq_len=[SEQ_LEN]):

class Dataset_Batch:
    def __init__(self, annotation_path, data_folder, mapping = {}):
        self.annotation_path = annotation_path
        self.data_folder = data_folder
        self.mapping = mapping


def save_unscaled_spot_features(spots_df, output_prefix=""):
    # extract unscaled features and save to CSV
    unscaled_df = spots_df[["PREFIX", "TRACK_ID", "FRAME", "LABEL"] + features].copy()
    if output_prefix != "":
        output_prefix = "_" + output_prefix

    out_path = os.path.join(GENERATED_DIR, f"unscaled_spot_features{output_prefix}.csv")
    unscaled_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved unscaled spot features to: {out_path}")

def save_unscaled_track_features(tracks_df, datasets, output_prefix=""):
    # extract unscaled track features and save to CSV
    merged_mapping = {}

    for label in datasets:
        overlap = merged_mapping.keys() & datasets[label].mapping.keys()  # intersection of keys
        if overlap:
            raise ValueError(f"Overlapping keys found when merging: {overlap}")
        merged_mapping.update(datasets[label].mapping)

    def match_label(prefix, merged_mapping):
        prefix = "_".join(prefix.split("_")[:2])
        return merged_mapping.get(prefix, np.nan)

    tracks_df["LABEL"] = tracks_df["PREFIX"].apply(match_label, args=(merged_mapping,))
    unscaled_df = tracks_df[["PREFIX", "TRACK_ID", "LABEL"] + track_features].copy()

    if output_prefix != "":
        output_prefix = "_" + output_prefix

    out_path = os.path.join(GENERATED_DIR, f"unscaled_track_features{output_prefix}.csv")

    unscaled_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved unscaled track features to: {out_path}")




# === Step 1: Load Annotations ===
def load_annotations(path, folder):

    folder = folder.split("/")[-1]
    match folder:
        case "CART":
            df = pd.read_excel(path, sheet_name="Summary")
            id_series = df.iloc[:, 1].astype(str).str.strip()
            label_series = df.iloc[:, 2].astype(float)
        case "2ND":
            df = pd.read_excel(path, sheet_name=0)
            id_series = df["Meso IL18 CAR T cells"].astype(str).str.strip()
            label_series = df["Labels"].astype(float)
        case "PDO":
            df = pd.read_excel(path, sheet_name="Statistics")
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "")
            id_series = df["Name"].astype(str).str.replace(" ", "")
            label_series = df["Score"]
        case "CAF":
            df = pd.read_excel(path, sheet_name="Statistics")
            df["Device Name"] = df["Name"].astype(str).str.replace(" ", "")
            id_series = df["Name"].astype(str).str.replace(" ", "")
            label_series = df["Score"]

        case _:
            raise Exception(f"Undefined Data loading type: {folder}. Specify how to load the annotations in load_annoations.")
            
    mapping = dict(zip(id_series, label_series))
    mapping = {f"{folder}_{device}": label for device, label in mapping.items()}
    print(f"Loaded annotation from {path}")
    print("Total entries:", len(mapping))
    print(mapping)
    return mapping


# === Step 2: Load Track/Spot Files ===
def load_tracks_and_spots(datasets):
    spots = []
    tracks = []

    for label in datasets:

        for file_name in os.listdir(datasets[label].data_folder):
            if not file_name.endswith("_tracks.csv"):
                continue
            
            prefix = file_name.replace("_tracks.csv", "")
            spot_file = file_name.replace("_tracks.csv", "_spots.csv")

            track_path = os.path.join(datasets[label].data_folder, file_name)
            spot_path = os.path.join(datasets[label].data_folder, spot_file)

            label_dict = datasets[label].mapping
            prefix_split = prefix.split("_")

            folder_name = datasets[label].data_folder.split("/")[-1]
            
            
            # Check if any split contains the annotation names instead of just the first
            index = 0
            original_prefix = prefix
            while index < len(prefix_split):
                if folder_name + "_" + prefix_split[index] in label_dict:
                    prefix = folder_name + "_" + prefix
                    label_prefix = folder_name + "_" + prefix_split[index]
                    break

                prefix = prefix.replace(prefix_split[index]+"_","")
                index += 1
            
            if index == len(prefix_split):
                raise Exception(f"{original_prefix} cannot be found in annotations file.")

            try:
                df_raw_track = pd.read_csv(track_path, encoding='latin1',
                                        header=None)
                names_track = df_raw_track.iloc[0].tolist()
                df_track = pd.read_csv(track_path, encoding='latin1',
                                    skiprows=4, names=names_track)
                #all as float
                df_track = df_track.apply(pd.to_numeric, errors='coerce')

                df_track['PREFIX'] = prefix
                df_track['LABEL'] = label_dict[label_prefix]

                df_raw_spot = pd.read_csv(spot_path, encoding='latin1',
                                        header=None)
                names_spot = df_raw_spot.iloc[0].tolist()
                df_spot = pd.read_csv(spot_path, encoding='latin1',
                                    skiprows=4, names=names_spot)
                df_spot = df_spot.apply(pd.to_numeric, errors='coerce')
                df_spot['PREFIX'] = prefix
                df_spot['LABEL'] = label_dict[label_prefix]

                tracks.append(df_track)
                spots.append(df_spot)

            except Exception as e:
                import sys, traceback
                print(label_dict)
                print(f"Failed to load {prefix}: {e}")
                traceback.print_exc()
                sys.exit(1)
                
                continue


    spots_df = pd.concat(spots, ignore_index=True)
    tracks_df = pd.concat(tracks, ignore_index=True)

    return spots_df, tracks_df


# === Step 3: Filter Valid Trajectories ===
def filter_valid_trajectories(spots_df, tracks_df, min_frames=10):
    
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames
                          ][["PREFIX", "TRACK_ID"]]
    spots_df_filtered = spots_df.merge(valid_ids, 
                                       on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_df_filtered = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]


    return spots_df_filtered, tracks_df_filtered

def compute_msd(x, y, max_lag=None):
    N = len(x)
    if max_lag is None:
        max_lag = N // 4
    msd = []
    for dt in range(1, max_lag + 1):
        dx = x[dt:] - x[:-dt]
        dy = y[dt:] - y[:-dt]
        squared_displacement = dx**2 + dy**2

        if len(squared_displacement) == 0:
            msd.append(0)
        else:
            msd.append(squared_displacement.mean())
    return msd

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
    
    spots_df["MEAN_SQUARE_DISPLACEMENT"] = 0
    all_msd = []

    index = 0
    total_tracks = len(spots_df.groupby(["PREFIX", "TRACK_ID"]))

    for (prefix, track_id), group in spots_df.groupby(["PREFIX", "TRACK_ID"]):
        print(f"Current Progress: {index/total_tracks*100:.2f}%")
        group = group.sort_values("FRAME")
        x = group["POSITION_X"].values
        y = group["POSITION_Y"].values
        frames = group["FRAME"].values

        msd = compute_msd(x, y, max_lag=(len(x)-1))

        # align MSD with the correct FRAME (starting at lag = 1)
        lag_frames = group["FRAME"].iloc[1:].values
        for frame_val, m in zip(lag_frames, msd):
            all_msd.append({
                "PREFIX": prefix,
                "TRACK_ID": track_id,
                "FRAME": frame_val,
                "MEAN_SQUARE_DISPLACEMENT": m,
            })

        index += 1

    msd_df = pd.DataFrame(all_msd)

    # Add the calculated MSD column to spots_df.
    # There is now an intialized MSD and MSD_new
    spots_df = spots_df.merge(
        msd_df,
        on=["PREFIX", "TRACK_ID", "FRAME"],
        how="left",
        suffixes=("", "_new")
    )

    # Combine original, initialized MSD with new calculated MSD if possible, otherwise use the initialized value.
    # Done to avoid NaN when merging because calculated MSD does not contain MSD(0), resuting in NaN for the first entry when merging.
    spots_df["MEAN_SQUARE_DISPLACEMENT"] = spots_df["MEAN_SQUARE_DISPLACEMENT_new"].combine_first(
        spots_df["MEAN_SQUARE_DISPLACEMENT"]
    )
    spots_df = spots_df.drop(columns=["MEAN_SQUARE_DISPLACEMENT_new"])

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

    for (p, tid), traj in spots_df.groupby(["PREFIX", "TRACK_ID"]):
        feat = traj[features].values
        if len(feat) >= seq_len:
            feat = feat[:seq_len]
        else:
            pad = np.zeros((seq_len - len(feat), len(features)))
            feat = np.vstack([feat, pad])

        feat_scaled = feat.copy()
        # df_temp = pd.DataFrame(feat, columns=features)
        # feat_scaled = scaler.transform(df_temp)

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
        f"[Save] Dataset saved: {GENERATED_DIR}/{output_prefix}_{seq_len}.npz & .csv "
        f"| Shape: {X.shape}"
    )



# === Step 6: Save Track-Level Dataset ===
def build_track_level_dataset(tracks_df, datasets,
                              output_prefix="", 
                              track_features = track_features):
    if len(track_features) == 0:
        print("[INFO] No track features available.")
        return
    
    merged_mapping = {}

    for label in datasets:
        overlap = merged_mapping.keys() & datasets[label].mapping.keys()  # intersection of keys
        if overlap:
            raise ValueError(f"Overlapping keys found when merging: {overlap}")
        merged_mapping.update(datasets[label].mapping)

    def match_label(prefix, merged_mapping):
        prefix = "_".join(prefix.split("_")[:2])
        return merged_mapping.get(prefix, np.nan)
    
    tracks_df["LABEL"] = tracks_df["PREFIX"].apply(match_label, args=(merged_mapping,))

    df = tracks_df.dropna(subset=track_features +
                          ["LABEL", "PREFIX", "TRACK_ID"]).copy()

    records = []
    
    for prefix, group in df.groupby("PREFIX"):
        group_scaled = group[track_features].values

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

    if output_prefix != "":
        output_prefix = "_" + output_prefix

    df_final.to_csv(f"{GENERATED_DIR}/{output_prefix}track_dataset.csv",
                    index=False)

    np.savez(f"{GENERATED_DIR}/{output_prefix}track_dataset.npz", 
            X=df_final[track_features].values, 
            y=df_final["LABEL"].values,
            track_ids=df_final[["PREFIX", "TRACK_ID"]].values)
    print(f"[Save] Dataset saved: {GENERATED_DIR}/{output_prefix}track_dataset.csv & .npz")


def filter_outer(spots_df):
    init_rows = len(spots_df)
    minor_zero_count = (spots_df['ELLIPSE_MINOR'] == 0).sum()
    aspect_neg_or_zero = (spots_df['ELLIPSE_ASPECTRATIO'] <= 0).sum()
    aspect_too_large = (spots_df['ELLIPSE_ASPECTRATIO'] > 5).sum()

    print(f"[Debug] Total rows before filtering: {init_rows}")
    print(f"[Debug] Rows with ELLIPSE_MINOR == 0: {minor_zero_count}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO <= 0: {aspect_neg_or_zero}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO > 5: {aspect_too_large}")

    rows_to_remove = (
        (spots_df['ELLIPSE_MINOR'] <= 0) |
        (spots_df['ELLIPSE_ASPECTRATIO'] <= 0) |
        (spots_df['ELLIPSE_ASPECTRATIO'] >= 5)
    )

    # Find all groups (PREFIX, TRACK_ID) where ANY row meets the condition
    bad_groups = spots_df.loc[rows_to_remove, ['PREFIX', 'TRACK_ID']].drop_duplicates()

    # Filter out those groups entirely
    filtered_df = spots_df.merge(bad_groups, on=['PREFIX', 'TRACK_ID'], how='left', indicator=True)
    filtered_df = filtered_df[filtered_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    spots_df = filtered_df
    
    minor_zero_count = (spots_df['ELLIPSE_MINOR'] == 0).sum()
    aspect_neg_or_zero = (spots_df['ELLIPSE_ASPECTRATIO'] <= 0).sum()
    aspect_too_large = (spots_df['ELLIPSE_ASPECTRATIO'] > 5).sum()


    print(f"[Debug] Total rows before filtering: {init_rows}")
    print(f"[Debug] Rows with ELLIPSE_MINOR == 0: {minor_zero_count}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO <= 0: {aspect_neg_or_zero}")
    print(f"[Debug] Rows with ELLIPSE_ASPECTRATIO > 5: {aspect_too_large}")
    filtered_rows = len(spots_df)
    removed_count = init_rows - filtered_rows
    print(f"[Filter] Removed {removed_count} invalid rows | Remaining: {filtered_rows}")

    num_groups_removed = bad_groups.shape[0]
    print(f"Removed {num_groups_removed} cell tracks.")

    return spots_df

def create_new_test_dataset(folder_path, annotation_path,
                        output_prefix="test", seq_len=20, min_frames=10):

    from Config import features, track_features, GENERATED_DIR
    print(track_features)

    # === 1. Load annotations ===
    device_to_score = load_annotations(f"{DATA_DIR}/PDO_annotation.xlsx",
                                    annotation_type="PDO")
    
    all_spots, all_tracks = [], []

    # === 2. Load files ===
    for fname in os.listdir(folder_path):
        if "_tracks.csv" in fname:
            prefix = fname.replace("_tracks.csv", "")
            device_name = prefix.split("_")[0].replace(" ", "").lower()
            if device_name not in device_to_score:
                print(f"[Warning] Device '{device_name}' not in annotation.")
                continue

            label = device_to_score[device_name]
            track_path = os.path.join(folder_path, fname)
            spot_path = os.path.join(folder_path, f"{prefix}_spots.csv")

            try:
                # load track
                df_raw_track = pd.read_csv(track_path, encoding='latin1', header=None)
                names_track = df_raw_track.iloc[0].tolist()
                df_track = pd.read_csv(track_path, encoding='latin1', skiprows=4, names=names_track)
                df_track["PREFIX"] = prefix
                df_track["LABEL"] = label
                all_tracks.append(df_track)

                # load spot
                df_raw_spot = pd.read_csv(spot_path, encoding='latin1', header=None)
                names_spot = df_raw_spot.iloc[0].tolist()
                df_spot = pd.read_csv(spot_path, encoding='latin1', skiprows=4, names=names_spot)
                df_spot["PREFIX"] = prefix
                df_spot["LABEL"] = label
                all_spots.append(df_spot)
            except Exception as e:
                print(f"[Error] {prefix}: {e}")

    if len(all_spots) == 0 or len(all_tracks) == 0:
        print("[Error] No valid files loaded.")
        return

    spots_df = pd.concat(all_spots, ignore_index=True)
    tracks_df = pd.concat(all_tracks, ignore_index=True)

    print(f"[Info] Loaded: {len(spots_df)} spot rows | {len(tracks_df)} track rows")

    # === 3. Filter valid tracks (min_frames) ===
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames][["PREFIX", "TRACK_ID"]]
    spots_df = spots_df.merge(valid_ids, on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_df = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]

    # === 4. Compute Spot features ===
    spots_df = spots_df.sort_values(by=["PREFIX", "TRACK_ID", "FRAME"])
    spots_df["VELOCITY_X"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_X"].diff().fillna(0)
    spots_df["VELOCITY_Y"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_Y"].diff().fillna(0)
    spots_df["SPEED"] = np.sqrt(spots_df["VELOCITY_X"]**2 + spots_df["VELOCITY_Y"]**2)
    spots_df["DIRECTION"] = np.arctan2(spots_df["VELOCITY_Y"], spots_df["VELOCITY_X"]) / np.pi

    spots_df["MEAN_SQUARE_DISPLACEMENT"] = 0

    total_tracks = len(spots_df.groupby(["PREFIX", "TRACK_ID"]))
    index = 0

    for (prefix, track_id), group in spots_df.groupby(["PREFIX", "TRACK_ID"]):
        print(f"Current Progress: {index/total_tracks*100:.2f}%")
        group = group.sort_values("FRAME")
        x = group["POSITION_X"].values
        y = group["POSITION_Y"].values
        frames = group["FRAME"].values

        
        msd = compute_msd(x, y, max_lag=(len(x)-1))

        for i in range(len(group)):
            if i >= 1:  # MSD is defined for lag >= 1
                mask = (
                    (spots_df["PREFIX"] == prefix)
                    & (spots_df["TRACK_ID"] == track_id)
                    & (spots_df["FRAME"] == group["FRAME"].iloc[i])
                )
                spots_df.loc[mask, "MEAN_SQUARE_DISPLACEMENT"] = msd[i - 1]
        index += 1

    drop_cols = [col for col in spots_df.columns if "INTENSITY" in col or col in ["POSITION_X", "POSITION_Y"]]
    spots_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    for f in features:
        spots_df[f] = pd.to_numeric(spots_df[f], errors='coerce')
    spots_df = spots_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    filter_outer(spots_df=spots_df)

    # === 5. Save Spot sequences ===
    X_list, y_list, track_id_list, rows = [], [], [], []
    for prefix, group_df in spots_df.groupby("PREFIX"):
        scaler = StandardScaler()
        scaler.fit(group_df[features])

        for (p, tid), traj in group_df.groupby(["PREFIX", "TRACK_ID"]):
            feat = traj[features].copy()
            if len(feat) >= seq_len:
                feat = feat.iloc[:seq_len]
            else:
                pad = pd.DataFrame(np.zeros((seq_len - len(feat), len(features))),
                                columns=features)
                feat = pd.concat([feat, pad], ignore_index=True)
            feat_scaled = scaler.transform(feat)
            X_list.append(feat_scaled)
            y_list.append(traj["LABEL"].iloc[0])
            track_id_list.append((p, tid))

            for t in range(seq_len):
                rows.append([f"{p}_{tid}", t] + list(feat_scaled[t]))

    X = np.array(X_list)
    y = np.array(y_list)
    track_ids = np.array(track_id_list, dtype=object)

    df_seq = pd.DataFrame(rows, columns=["SampleID", "Frame"] + features)
    df_seq.to_csv(f"{GENERATED_DIR}/{output_prefix}_trajectory_{seq_len}.csv", index=False)
    np.savez(f"{GENERATED_DIR}/{output_prefix}_trajectory_{seq_len}.npz", X=X, y=y, track_ids=track_ids)

    print(f"[Save] Trajectory-level test set saved | Shape: {X.shape}")

    # === 6. Save Track-level features ===
    records = []
    for prefix, group in tracks_df.groupby("PREFIX"):
        scaler = StandardScaler()
        
        group_feat = group[track_features].dropna().values
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

    df_track = pd.DataFrame(records)

    df_track.to_csv(f"{GENERATED_DIR}/{output_prefix}_track.csv", index=False)
    np.savez(f"{GENERATED_DIR}/{output_prefix}_track.npz",
             X=df_track[track_features].values,
             y=df_track["LABEL"].values,
             track_ids=df_track[["PREFIX", "TRACK_ID"]].values)
    print(f"[Save] Track-level test set saved | Shape: {df_track.shape}")

    # === 7. Save unscaled version ===
    spots_df.to_csv(f"{GENERATED_DIR}/{output_prefix}_test_unscaled_spot_features.csv", index=False)
    tracks_df.to_csv(f"{GENERATED_DIR}/{output_prefix}_test_unscaled_track_features.csv", index=False)
    print(f"[Save] Raw features saved.")

if __name__ == "__main__":
    create_training_dataset = True
    create_test_dataset = False

    if create_training_dataset:
        datasets = {
            "CART": Dataset_Batch(annotation_path = f"{DATA_DIR}/CART annotations.xlsx", 
                                  data_folder = f"{DATA_DIR}/CART"),

            "2nd": Dataset_Batch(annotation_path = f"{DATA_DIR}/2nd batch annotations.xlsx",
                                 data_folder = f"{DATA_DIR}/2ND"),
            
            "PDO": Dataset_Batch(annotation_path = f"{DATA_DIR}/PDO_annotation.xlsx",
                                 data_folder = f"{DATA_DIR}/PDO"),
        }

        # datasets = {
        #     "CAF": Dataset_Batch(annotation_path = f"{DATA_DIR}/CAF_annotations.xlsx",
        #                          data_folder = f"{DATA_DIR}/CAF"),
        # }

        for label in datasets:
            datasets[label].mapping = load_annotations(datasets[label].annotation_path, datasets[label].data_folder)
        
        spots_df, tracks_df = load_tracks_and_spots(datasets)

        # Create train dataset
        spots_df, tracks_df = filter_valid_trajectories(spots_df, tracks_df)
        spots_df = compute_features(spots_df)
        spots_df = filter_outer(spots_df)
        save_unscaled_spot_features(spots_df, output_prefix="")
        save_unscaled_track_features(tracks_df, datasets=datasets, output_prefix="")

        all_features = sorted(set(features).union(set(track_features)))
        spots_aligned = spots_df.reindex(columns=all_features, fill_value=0)
        tracks_aligned = tracks_df.reindex(columns=all_features, fill_value=0)

        combined = pd.concat([spots_aligned, tracks_aligned], axis=0)

        # Fit scaler on global data
        scaler = StandardScaler()
        scaler.fit(combined)

        # Transform back into DataFrames
        spots_scaled = pd.DataFrame(
            scaler.transform(spots_aligned),
            columns=all_features,
            index=spots_df.index
        )
        tracks_scaled = pd.DataFrame(
            scaler.transform(tracks_aligned),
            columns=all_features,
            index=tracks_df.index
        )

        # Restore original columns
        spots_scaled = spots_scaled[features]
        tracks_scaled = tracks_scaled[track_features]
        spots_final = spots_df.copy()
        spots_final[features] = spots_scaled

        tracks_final = tracks_df.copy()
        tracks_final[track_features] = tracks_scaled

        from Config import SEQ_LEN
        for seq_len_iter in [SEQ_LEN]:
            align_and_save_dataset(spots_final,
                                features, seq_len=seq_len_iter,
                                output_prefix="trajectory_dataset")
        
        build_track_level_dataset(tracks_final, datasets=datasets, output_prefix="")   


        
        print("Dataset creation completed.")
        # Create test dataset
    if create_test_dataset:
        create_new_test_dataset(
        folder_path=f"{DATA_DIR}/PDO",
        annotation_path=f"{DATA_DIR}/PDO_annotation.xlsx",
        output_prefix="cart_test",
        seq_len=SEQ_LEN)
