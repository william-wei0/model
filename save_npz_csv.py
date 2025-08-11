import numpy as np
import pandas as pd

# Load the .npz file
data = np.load(r"C:\Users\billy\Downloads\training_logs_unified.npz")

# Show what arrays are stored in the .npz file
print("Arrays in .npz file:", data.files)

# Save each array to a CSV
for key in data.files:
    array = data[key]
    
    # If it's 1D, turn it into a 2D column for CSV
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    
    # Save to CSV using pandas (for cleaner formatting)
    df = pd.DataFrame(array)
    df.to_csv(fr'C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Generated\{key}.csv', index=False)

    print(f"Saved {key}.csv")

