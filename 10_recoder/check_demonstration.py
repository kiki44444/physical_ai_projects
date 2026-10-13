import numpy as np
import os

path = "datasets/demonstrations/episode_000.npz"

print("File exists:", os.path.exists(path))
print("File size:", os.path.getsize(path), "bytes")

with np.load(path, allow_pickle=False) as data:
    print("Keys:", data.files)
    print("Number of keys:", len(data.files))

    for key in data.files:
        print(key, data[key].shape, data[key].dtype)