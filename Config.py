features = [
    'RADIUS', 'AREA', 'PERIMETER', 'CIRCULARITY',
    'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_ASPECTRATIO',
    'SOLIDITY', 'SPEED'
]
track_features = [
    "TRACK_DURATION", "TRACK_DISPLACEMENT", "TRACK_MEAN_SPEED",
    "TRACK_MAX_SPEED", "TRACK_MIN_SPEED", "TRACK_STD_SPEED",
    "TOTAL_DISTANCE_TRAVELED", "MAX_DISTANCE_TRAVELED", "CONFINEMENT_RATIO",
    "MEAN_STRAIGHT_LINE_SPEED", "LINEARITY_OF_FORWARD_PROGRESSION",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]
FEATURE_LEN = len(features)
TRACK_LEN = len(track_features)
SEQ_LEN = 360 # Choose from 20, 100, 360
# Run Step1 if you haven't yet to generate the dataset, or any changes to the dataset.
# Run Step2 once to train the track model.
# Change SEQ_LEN to 20, 100, and 360, and run Step 3, 4 and 5 under each SEQ_LEN.