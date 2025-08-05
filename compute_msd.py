def compute_msd(x, y, max_lag=None):
    x = np.asarray(x)
    y = np.asarray(y)

    N = len(x)
    if max_lag is None:
        max_lag = N // 4
    msd = []
    for dt in range(1, max_lag + 1):
        dx = x[dt:] - x[:-dt]
        dy = y[dt:] - y[:-dt]
        squared_displacement = dx**2 + dy**2
        msd.append(squared_displacement.mean())
    return msd

import pandas as pd
import numpy as np
data = {
    "x": [0, 1, 2, 3],
    "y":[0, 0, 1, 2]
}
df = pd.DataFrame(data)


print(compute_msd(df["x"], df["y"], 3))