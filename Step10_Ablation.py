if __name__ == "__main__":

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    from Config import DATA_DIR, MODEL_DIR, RESULTS_DIR, SEQ_LEN, GENERATED_DIR, HIDDEN_SIZE_LSTM, DROPOUT
    from Step1_data import (load_annotations,load_tracks_and_spots,
                            filter_valid_trajectories, compute_features,
                            align_and_save_dataset, build_track_level_dataset, 
                            create_new_test_dataset)
    from Step8_unified_fusion import Train_UnifiedFusionModel, Test_UnifiedFusionModel
    from Step9_SHAP_Unified import SHAP_UnifiedFusionModel

    # === features and Config ===
    all_features = [  # time-based
        'RADIUS', 'AREA', 'PERIMETER', 'CIRCULARITY',
        'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_ASPECTRATIO', 'SOLIDITY', 'SPEED', 'MEAN_SQUARE_DISPLACEMENT'
    ]

    all_track_features = [  # track-based
        "TRACK_DURATION", "TRACK_DISPLACEMENT", "TRACK_MEAN_SPEED",
        "TRACK_MAX_SPEED", "TRACK_MIN_SPEED", "TRACK_STD_SPEED",
        "TOTAL_DISTANCE_TRAVELED", "MAX_DISTANCE_TRAVELED", "CONFINEMENT_RATIO",
        "MEAN_STRAIGHT_LINE_SPEED", "LINEARITY_OF_FORWARD_PROGRESSION",
        "MEAN_DIRECTIONAL_CHANGE_RATE"
    ]

    ablation_configs = {
        # "baseline": {
        #     "features": all_features,
        #     "track_features": all_track_features
        # },
        # "remove_negative": {
        #     "features": all_features,
        #     "track_features": [
        #         f for f in all_track_features if f not in [
        #             "TRACK_MEAN_SPEED", "TRACK_STD_SPEED", "MEAN_STRAIGHT_LINE_SPEED"
        #         ]
        #     ]
        # },
        # "top6_shap": {
        #     "features": ["RADIUS", "SOLIDITY", "ELLIPSE_MINOR"],
        #     "track_features": ["TRACK_DURATION", "TRACK_STD_SPEED", "TRACK_MEAN_SPEED"]
        # }
        "Specify" : {
            "features": ['AREA', 'PERIMETER', 'CIRCULARITY', 
                        'ELLIPSE_ASPECTRATIO', 'SOLIDITY', 'SPEED', 'MEAN_SQUARE_DISPLACEMENT'],
            "track_features" :["TRACK_STD_SPEED", "TRACK_DISPLACEMENT",
                                "MEAN_DIRECTIONAL_CHANGE_RATE"]
        },

    }


    results_summary = []

    # === begin experiment ===
    for name, cfg in ablation_configs.items():
        print(f"\n===== Running Ablation: {name} =====")

        # model and dataset save route
        prefix = f"ablation_{name}"
        seq_path = os.path.join(GENERATED_DIR, f"{prefix}_{SEQ_LEN}.npz")
        track_path = os.path.join(GENERATED_DIR, f"{prefix}track_dataset.npz")

        model_path = os.path.join(RESULTS_DIR, f"{prefix}/model_{prefix}_.pth")
        result_path = os.path.join(RESULTS_DIR, prefix)
        os.makedirs(result_path, exist_ok=True)


        create_new_dataset = False
        train_model = True


        # step 1: create dataset
        if (create_new_dataset):
            print("Creating New Dataset...")
            cart_labels = load_annotations(f"{DATA_DIR}/CART annotations.xlsx",
                                        annotation_type="CART")
            second_labels = load_annotations(f"{DATA_DIR}/2nd batch annotations.xlsx",
                                            annotation_type="2nd batch")
            
            spots_df, tracks_df = load_tracks_and_spots(
                folder=f"{DATA_DIR}/TRACK",
                cart_labels=cart_labels,
                second_labels=second_labels
            )
            
            spots_df, tracks_df = filter_valid_trajectories(spots_df, tracks_df)
            spots_df = compute_features(spots_df)

            align_and_save_dataset(spots_df,
                                    cfg["features"], seq_len=SEQ_LEN,
                                    output_prefix=prefix)
            
            build_track_level_dataset(tracks_df, cart_labels, second_labels,
                                    prefix, cfg["track_features"])    
            print("Finished Creating New Training Dataset.")

            create_new_test_dataset(
            folder_path=f"{DATA_DIR}/PDO",
            annotation_path=f"{DATA_DIR}/PDO_annotation.xlsx",
            output_prefix="cart_test",
            seq_len=SEQ_LEN)
            print("Finished Creating New Test Dataset.")
            
        else:
            print("\nDid not create new dataset.\n")


        # Step2: training model
        seq_input_size = len(cfg["features"])
        track_input_size = len(cfg["track_features"])


        seq_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\trajectory_dataset_100.npz"
        track_path = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\track_dataset.npz"

        
        if train_model:
            min_pow_fusion = 4
            max_pow_fusion = 12

            min_pow_hidden = 2
            max_pow_hidden = 8

            train_accuracies_df = pd.DataFrame([["Train Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
            val_accuracies_df = pd.DataFrame([["Validation Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
            test_accuracies_df = pd.DataFrame([["Test Accuracies"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])
            r_squared_df = pd.DataFrame([["R-Squared (Test)"]+[2**i for i in range(min_pow_hidden, max_pow_hidden)]])

            for fusion_pow in range(min_pow_fusion, max_pow_fusion):
                fusion_size = 2**fusion_pow
                train_accuracies = [fusion_size]
                val_accuracies = [fusion_size]
                test_accuracies = [fusion_size]
                r_squareds = [fusion_size]

                for hidden_pow in range(min_pow_hidden, max_pow_hidden):
                    hidden_size = 2**hidden_pow
                    print(f"Hidden Size: {hidden_size} | Fusion Size {fusion_size}")

                    metrics = Train_UnifiedFusionModel(
                        seq_path=seq_path,
                        track_path=track_path,
                        model_save_path=model_path,
                        result_path=result_path,
                        seq_input_size=seq_input_size,
                        track_input_size=track_input_size,
                        hidden_size = hidden_size,
                        fusion_size = fusion_size,
                        dropout=DROPOUT,
                        test_prefix = prefix
                    )
                            # Step4: record result
                    results_summary.append({
                        "config_name": name,
                        "best_train_acc": metrics["best_train_acc"],
                        "best_test_acc": metrics["best_test_acc"],
                        "f1_score": metrics["f1_score"],
                        "auc": metrics["auc"]
                    })

                    train_accuracies.append(metrics["best_train_acc"])
                    val_accuracies.append(metrics["best_val_acc"])
                    test_accuracies.append(metrics["best_test_acc"])
                    r_squareds.append(metrics["r2"])
                    
                
                train_accuracies_df.loc[len(train_accuracies_df)] = train_accuracies
                val_accuracies_df.loc[len(val_accuracies_df)] = val_accuracies
                test_accuracies_df.loc[len(test_accuracies_df)] = test_accuracies
                r_squared_df.loc[len(r_squared_df)] = r_squareds

            empty_row = pd.DataFrame([[None] * train_accuracies_df.shape[1]], columns=train_accuracies_df.columns)
            all_accuracies_df = pd.concat([train_accuracies_df, empty_row, 
                                           val_accuracies_df, empty_row, 
                                           test_accuracies_df, empty_row, 
                                           r_squared_df], ignore_index=True)

            from datetime import datetime
            now = datetime.now()
            time_str = now.strftime("%H_%M")

            all_accuracies_df.to_csv(os.path.join(result_path, f"accuracies_{time_str}.csv"), index=False)

            

        # test_results_dir = os.path.join(result_path, "test_original")
        # os.makedirs(test_results_dir, exist_ok=True)

        # Test_UnifiedFusionModel(seq_path,
        #                         track_path,
        #                         model_path,
        #                         output_dir=result_path,
        #                         seq_input_size=seq_input_size, 
        #                         track_input_size=track_input_size,
        #                         hidden_size=HIDDEN_SIZE_LSTM,
        #                         dropout=DROPOUT,)
        
        # test_seq_path = os.path.join(GENERATED_DIR, "cart_test_trajectory_100.npz")
        # test_track_path = os.path.join(GENERATED_DIR, "cart_test_track.npz")
        # test_results_dir = os.path.join(result_path, "test_second")
        # os.makedirs(test_results_dir, exist_ok=True)

        # Test_UnifiedFusionModel(test_seq_path,
        #                         test_track_path,
        #                         model_path,
        #                         output_dir=test_results_dir,
        #                         seq_input_size=seq_input_size, 
        #                         track_input_size=track_input_size,
        #                         hidden_size=HIDDEN_SIZE_LSTM,
        #                         dropout=DROPOUT,)
        

        # # Step3: shap analysis
        # SHAP_UnifiedFusionModel(
        #     seq_length=SEQ_LEN,
        #     features=cfg["features"],
        #     track_features=cfg["track_features"],
        #     model_save_path=model_path,
        #     result_path=result_path,
        #     seq_path=seq_path,
        #     track_path=track_path
        # )


    # === show and save comparison sheet ===
    df = pd.DataFrame(results_summary)
    df = df.sort_values(by="best_train_acc", ascending=False)
    print("\n=== Summary of Ablation Results ===")
    print(df)

    summary_path = os.path.join(RESULTS_DIR, f"ablation_summary_{SEQ_LEN}.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


    # === save plot ===
    plt.figure(figsize=(10, 5))
    df.plot(x='config_name', y=['best_train_acc', 'f1_score', 'auc'], kind='bar')
    plt.title(f"Ablation Performance Comparison (SeqLen={SEQ_LEN})")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"ablation_barplot_{SEQ_LEN}.png"))
    plt.close()
    print("Saved performance bar plot.")