import sif_parser
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

root_folder = "DP"
labeled_data = {}
M = 250
N = 1024
shape = np.array([M, N])

pixels = np.arange(0, N)

lambs = 0.037033476478940355 * pixels + 381.02184936261204 + 0.21
dlamb = lambs[1] - lambs[0]

i_min = 512
i_max = 600

for file in os.listdir(root_folder):

    try:
        file_path = root_folder + "/" + file
        label = file.split("_")[1]

        if label == "data.csv":
            continue

        if label == "blank":
            continue

        # Read sif file and average where spectral lines exist
        data, info = sif_parser.np_open(file_path)
        data_mean = data[:, i_min:i_max].mean(axis = 1)

        # Labeling, this is very lazy as it only uses the first instance of a concentration. However, it may be nice to just verify on the other samples
        if label not in labeled_data.keys():
            labeled_data[label] = data_mean.flatten()

        else:
            continue

        scale = np.sum(data_mean.mean(axis = 0) * dlamb)

        fig, ax = plt.subplots(1)
        ax.set_title(label)
        ax.plot(lambs, data_mean.mean(axis = 0) / scale)
        ax.grid(True)

        plt.show()
            

    # Things break and I can't be bothered
    except Exception as e:
        print(f"Error: {e}")
        pass


# Output data
df = pd.DataFrame(labeled_data)
df.to_csv(root_folder + "/cnn_labeled_data.csv", index = False)
np.save(root_folder + "/shape.npy", shape)
