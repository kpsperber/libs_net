from LIBS_Net import LIBS_Net

net = LIBS_Net()

# Reads in the processed data
net.read_data("DP")

# Splits up the data into test and training
net.split_data()

# Initializes the model
net.build_model()

# Trains and evaluates the model
history = net.train()
results = net.evaluate()

net.save()