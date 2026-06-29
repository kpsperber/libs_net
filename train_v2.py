from LIBS_Net_v2 import LIBS_Net
import os

net = LIBS_Net()
root_folder = "DP"

global_file = os.path.join(net.root, net.global_file)
local_file = os.path.join(net.root, net.local_file)
physics_file = os.path.join(net.root, net.physics_file)

if not os.path.exists(global_file):
    net.read_global_data(root_folder)
    net.split_global_data()
    net.build_global_model()
    net.train_global()
    net.evaluate_global(root_folder)
    net.save_global()

else:
    net.load_global()

if not os.path.exists(local_file):
    net.read_local_data(root_folder)
    net.split_local_data()
    net.build_local_model()
    net.train_local()
    net.evaluate_local(root_folder)
    net.save_local()

else:
    net.load_local()

    if not os.path.exists(physics_file):
        net.read_physics_data(root_folder)

