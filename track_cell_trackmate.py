import imagej
import pandas as pd
from pathlib import Path


# Initialize ImageJ with Fiji
ij = imagej.init("sc.fiji:fiji", headless=True)


# Import Java classes
TrackMate = ij.py.get_jvm().jimport('fiji.plugin.trackmate.TrackMate')
Model = ij.py.get_jvm().jimport('fiji.plugin.trackmate.Model')
Settings = ij.py.get_jvm().jimport('fiji.plugin.trackmate.Settings')
LogLogger = ij.py.get_jvm().jimport('fiji.plugin.trackmate.util.LogLogger')
ThresholdDetectorFactory = ij.py.get_jvm().jimport('fiji.plugin.trackmate.detection.ThresholdDetectorFactory')
SimpleLAPTrackerFactory = ij.py.get_jvm().jimport('fiji.plugin.trackmate.tracking.sparselap.SimpleLAPTrackerFactory')

def run_trackmate(image_path, output_folder):
    # Load image
    imp = ij.io().open(str(image_path))
    
    # --- Compute auto-threshold for this image ---
    # Duplicate the image to avoid modifying the original
    imp_dup = imp.duplicate()
    from ij import IJ
    IJ.setAutoThreshold(imp_dup, "Otsu dark")
    threshold_value = imp_dup.getProcessor().getMinThreshold()

    # Settings
    settings = Settings(imp)
    settings.detectorFactory = ThresholdDetectorFactory()
    settings.detectorSettings = {
        'INTENSITY_THRESHOLD': threshold_value,   # set the auto threshold
        'DO_SUBPIXEL_LOCALIZATION': True,
        'RADIUS': 5.0
    }

    # Tracker settings remain the same
    settings.trackerFactory = SimpleLAPTrackerFactory()
    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
    settings.trackerSettings['LINKING_MAX_DISTANCE'] = 15.0
    settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 15.0
    settings.trackerSettings['MAX_FRAME_GAP'] = 2

    # Model + logger
    model = Model()
    model.setLogger(LogLogger())

    # Run TrackMate
    trackmate = TrackMate(model, settings)
    if not trackmate.checkInput() or not trackmate.process():
        print("TrackMate failed on", image_path)
        return

    # Continue with spots/tracks export...


    # ---- Spots ----
    spots = []
    for spot in model.getSpots().iterable(True):
        spots.append({
            'ID': spot.ID(),
            'FRAME': spot.getFeature('FRAME'),
            'X': spot.getFeature('POSITION_X'),
            'Y': spot.getFeature('POSITION_Y'),
            'Z': spot.getFeature('POSITION_Z'),
            'INTENSITY': spot.getFeature('MEAN_INTENSITY')
        })
    pd.DataFrame(spots).to_csv(output_folder / f"{image_path.stem}_spots.csv", index=False)

    # ---- Tracks ----
    tracks = []
    for track_id in model.getTrackModel().trackIDs(True):
        track_spots = model.getTrackModel().trackSpots(track_id)
        for spot in track_spots:
            tracks.append({
                'TRACK_ID': track_id,
                'SPOT_ID': spot.ID(),
                'FRAME': spot.getFeature('FRAME'),
                'X': spot.getFeature('POSITION_X'),
                'Y': spot.getFeature('POSITION_Y'),
                'Z': spot.getFeature('POSITION_Z'),
                'INTENSITY': spot.getFeature('MEAN_INTENSITY')
            })
    pd.DataFrame(tracks).to_csv(output_folder / f"{image_path.stem}_tracks.csv", index=False)

    print(f"Saved results for {image_path}")

if __name__ == "__main__":
    root_folder = Path("C:/path/to/larger_folder")
    output_root = root_folder / "trackmate_results"
    output_root.mkdir(exist_ok=True)

    # Loop over all subfolders
    for tif_file in root_folder.rglob("*.tif"):
        subfolder = tif_file.parent.relative_to(root_folder)
        output_folder = output_root / subfolder
        output_folder.mkdir(parents=True, exist_ok=True)
        run_trackmate(tif_file, output_folder)
