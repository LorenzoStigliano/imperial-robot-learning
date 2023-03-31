import numpy as np
import torch
from matplotlib import pyplot as plt


# Turn on interactive mode for PyPlot, to prevent the displayed graph from blocking the program flow.
plt.ion()


# Define the neural network architecture.
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=4, out_features=32)
        self.layer_2 = torch.nn.Linear(in_features=32, out_features=32)
        #self.layer_3 = torch.nn.Linear(in_features=10, out_features = 10)
        self.output_layer = torch.nn.Linear(in_features=32, out_features=2)

    def forward(self, input_data):
        layer_1_output = torch.nn.functional.leaky_relu(self.layer_1(input_data))
        layer_2_output = torch.nn.functional.leaky_relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)

        return output


#Hyperparameters to test over:
#    learning_rate
#    epochs
#    layers
#    activation function
    
#learning_rate = [0.01, 0.001, 0.0001, 0.00001]
#epochs = [10, 25, 50, 75, 100]
#layers = [1, 2, 3, 4]
#hidden_layer_size = [4,8,12,16,32]
"""
# Initialise the network, loss function, and optimiser.
network = Network()
criterion = torch.nn.MSELoss()
optimiser = torch.optim.SGD(network.parameters(), lr=0.01)

# Create some dummy data.
dataset_inputs = torch.tensor(np.random.uniform(0, 1, [100, 4]).astype(np.float32))
dataset_labels = torch.tensor(np.zeros([100, 4], dtype=np.float32))
for i in range(100):
    dataset_labels[i, 0] = 2 * dataset_inputs[i, 0] + 0.5 + np.random.normal(0, 0.01)
    dataset_labels[i, 1] = dataset_inputs[i, 1] * dataset_inputs[i, 2] + np.random.normal(0, 0.01)
    dataset_labels[i, 2] = dataset_inputs[i, 3] * dataset_inputs[i, 3] + np.random.normal(0, 0.01)
    dataset_labels[i, 3] = 0.5 + np.random.normal(0, 0.01)

# Divide the data into training and validation subsets.
training_inputs = dataset_inputs[:80]
training_labels = dataset_labels[:80]
validation_inputs = dataset_inputs[80:]
validation_labels = dataset_labels[80:]

# Convert the data into tensors and create a DataLoader for each set.
training_dataset = torch.utils.data.TensorDataset(training_inputs, training_labels)
validation_dataset = torch.utils.data.TensorDataset(validation_inputs, validation_labels)
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=10, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=False)

# Training loop.
training_losses = []
validation_losses = []
for epoch in range(100):
    # Training phase.
    network.train()
    training_loss = 0
    for inputs, labels in training_loader:
        optimiser.zero_grad()
        predictions = network(inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimiser.step()
        training_loss += loss.item()
    training_loss /= len(training_loader)
    training_losses.append(training_loss)

    # Validation phase.
    network.eval()
    validation_loss = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            predictions = network(inputs)
            loss = criterion(predictions, labels)
            validation_loss += loss.item()
    validation_loss /= len(validation_loader)
    validation_losses.append(validation_loss)

    # Print the training and validation losses.
    print(f"[Epoch {epoch + 1}]\tTraining Loss: {training_loss:.4f}\tValidation Loss: {validation_loss:.4f}")

    # Plot and save the training and validation losses.
    plt.clf()
    plt.plot(range(epoch + 1), training_losses, label="Training loss", color=(0.8, 0.2, 0.2))
    plt.plot(range(epoch + 1), validation_losses, label="Validation loss", color=(0.2, 0.6, 0.2))
    plt.title("Loss Curve for Dummy Data")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.legend()
    plt.savefig("loss-curve.png")
    plt.pause(0.01)

"""
