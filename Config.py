features = [ # Time-based Features 
    # Morphological Features
    'RADIUS', 'AREA', 'PERIMETER', 'CIRCULARITY', # Geometric Properties
    'ELLIPSE_MAJOR', 'ELLIPSE_MINOR', 'ELLIPSE_ASPECTRATIO', # Ellipse-fitting-based features

    # Motion Features
    'SPEED', 'DIRECTION' # Calculated 
]

track_features = [ # Track-Level Statistics Features
    "TRACK_DURATION", "TRACK_DISPLACEMENT", "TRACK_MEAN_SPEED",
    "TRACK_MAX_SPEED", "TRACK_MIN_SPEED", "TRACK_STD_SPEED",
    "TOTAL_DISTANCE_TRAVELED", "MAX_DISTANCE_TRAVELED", "CONFINEMENT_RATIO",
    "MEAN_STRAIGHT_LINE_SPEED", "LINEARITY_OF_FORWARD_PROGRESSION",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]

SEQ_LEN = 20 # Choose from 20, 100, 360

FEATURE_LEN = len(features) # = 11
TRACK_LEN = len(track_features) # = 12
# Run Step1 if you haven't yet to generate the dataset, or any changes to the dataset.
# Run Step2 once to train the track model.
# Change SEQ_LEN to 20, 100, and 360, and run Step 3, 4 and 5 under each SEQ_LEN.