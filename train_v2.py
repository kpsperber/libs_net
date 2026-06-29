from LIBS_Net_v2 import LIBS_Net
import os

net = LIBS_Net()

global_file = os.path.join(net.root, net.global_file)
local_file = os.path.join(net.root, net.local_file)
physics_file = os.path.join(net.root, net.physics_file)

if not os.path.exists(global_file):
    net.read_global_data("DP")
    net.split_global_data()
    net.build_global_model()
    net.train_global()
    net.evaluate_global("DP")
    net.save_global()

else:
    net.load_global()

if not os.path.exists(local_file):
    net.read_local_data("DP")
    net.split_local_data()
    net.build_local_model()
    net.train_local()
    net.evaluate_local("DP")
    net.save_local()

else:
    net.load_local()

