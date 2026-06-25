import os

from LIBS_Net import LIBS_Net

net = LIBS_Net()
root = os.getcwd()

# Reads in the processed data
net.read_cnn_data("DP")

# Splits up the data into test and training
net.split_cnn_data()

# Initializes the individual models and trains separately. If the model exists it skips the training step
cnn_file = os.path.join(root, net.cnn_file)
physics_file = os.path.join(root, net.physics_file)

if os.path.exists(cnn_file):
    net.load_cnn()

else:
    net.build_cnn_model()
    net.split_cnn_data()
    net.train_cnn()
    net.evaluate_cnn()

if os.path.exists(physics_file):
    net.load_physics()

else:
    net.build_physics_model()
    net.split_physics_data()
    net.train_physics()

# Build full model and train


# Save everything
net.save_cnn()