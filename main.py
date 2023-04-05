# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


# Define BloodPressureDataset class for handling blood pressure data
class BloodPressureDataset(Dataset):
    def __init__(self, csv_file):
        # Load CSV file containing blood pressure data
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        # Return the number of data points in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get data and labels for a given index
        row = self.data.iloc[idx]
        parameters = torch.tensor([row['age'], row['weight'], row['heart_rate'], row['movement_level']],
                                  dtype=torch.float32)
        label = torch.tensor(row['blood_pressure'], dtype=torch.float32)
        return parameters, label


# Define BloodPressureModel class for the neural network model
class BloodPressureModel(nn.Module):
    def __init__(self):
        super(BloodPressureModel, self).__init__()
        # Define the fully connected layers
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # Define the forward pass through the neural network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the training and testing datasets
train_dataset = BloodPressureDataset("train_data.csv")
test_dataset = BloodPressureDataset("test_data.csv")

# Create DataLoaders for the training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the neural network model, loss function, and optimizer
model = BloodPressureModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for a specified number of epochs
num_epochs = 350
for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass to get predictions
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs.view(-1), labels)

        # Backward pass to calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

    # Print the loss for the current epoch (Prints 1, 10, 50, 100, 150, 200, 250, 300, 350)
    if (epoch + 1) in [10]:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    if (epoch + 1) == 1 or (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluate the model on the test dataset
model.eval()
test_loss = 0
with torch.no_grad():
    for features, labels in test_loader:
        # Forward pass to get predictions
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs.view(-1), labels)

        # Accumulate test loss
        test_loss += loss.item()

# Calculate and print the average test loss
print(f"Test Loss: {test_loss / len(test_loader)}")


# Define a function to calculate the accuracy rate based on Mean Absolute Error (MAE)
def calculate_accuracy(true_values, predicted_values):
    mae = np.mean(np.abs(true_values - predicted_values))
    max_val = np.max(true_values)  # You can also use a predetermined maximum possible blood pressure value
    accuracy = 1 - (mae / max_val)
    return accuracy * 100


# Use the trained model to predict blood pressure for the test dataset
model.eval()
predicted_values = []
true_values = []

with torch.no_grad():
    for features, labels in test_loader:
        # Forward pass to get predictions
        outputs = model(features)
        predicted_values.extend(outputs.view(-1).tolist())
        true_values.extend(labels.tolist())

predicted_values = np.array(predicted_values)
true_values = np.array(true_values)

# Calculate the accuracy rate
accuracy_rate = calculate_accuracy(true_values, predicted_values)
print(f"Accuracy Rate: {accuracy_rate:.2f}%")
