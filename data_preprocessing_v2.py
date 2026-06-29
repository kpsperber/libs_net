import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sif_parser
import sys

def remove_K_Ca(normalized_data):
    nd_data = np.zeros_like(normalized_data)

def normalize(data_1D):
    normalized_data = np.zeros_like(data_1D)
    return normalized_data

def fit_voigt_profile(nd_data):
    return []

root_folder = "DP"
training_folder = os.path.join(root_folder, "Training Data")
os.makedirs(training_folder, exist_ok = True)

M = 250
N = 1024

data_shape = np.array([M, N])
pixels = np.arange(0, N)
lambs = 0.037033476478940355 * pixels + 381.02184936261204 + 0.21
dlamb = lambs[1] - lambs[0]

i_min = 512
i_max = 600
lamb_min = 398
lamb_max = 404

global_cnn_data = {}
pinn_data = {}
local_cnn_data = {}

for file in os.listdir(root_folder):
    if not file.lower().endswith(".sif"):
        continue

    print(f"Preprocessing: {file}")
    file_path = os.path.join(root_folder, file) 

    # Extract Label, We will want to replace this with the real values
    label = file.split("_")[1]
    
    if label == "blank":
        label = "0ppm"

    # Read in the data and compress into 1D
    data_2D, info = sif_parser.np_open(file_path)
    data_1D = data_2D[:, i_min:i_max].mean(axis = 1)

    # Normalize
    normalized_data = normalize(data_1D) 

    # Save full spectrum for Global CNN
    global_cnn_data[label] = data_1D.flatten()

    # Background subtract K and Ca Peaks
    nd_data = remove_K_Ca(normalized_data)

    # Fit Voigt Profile to Nd Spectrum
    fit_parameters = fit_voigt_profile(nd_data)

    # Extract local data
    j_min = np.argmin(np.abs(lambs - lamb_min))
    j_max = np.argmin(np.abs(lambs - lamb_max))
    local_data = data_1D[:, j_min : j_max]
    lamb_local = lambs[j_min : j_max]

    data_shape_local = np.array([M, len(lamb_local)])

    local_cnn_data[label] = local_data.flatten()

df = pd.DataFrame(global_cnn_data)
df.to_csv(training_folder + "/cnn_global_data.csv", index = False)
np.save(training_folder + "/shape.npy", data_shape)
np.save(training_folder + "/lambs.npy", lambs)

df = pd.DataFrame(local_cnn_data)
df.to_csv(training_folder + "/cnn_local_data.csv", index = False)
np.save(training_folder + "/shape_local.npy", data_shape_local)








