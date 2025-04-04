import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter
from data.dataset import Dataset
from models.network import SimpleNN
from config import Config

# Initialize Weights and Biases
wandb.init(project="pytorch-simple-nn")

# Configuration
config = Config()
writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

# Load datasets
dataset = Dataset(num_points=10^4)  # Increased number of points
x_data, y_data = dataset.get_data()

# Split into train and test (80-20 split)
train_size = int(0.8 * len(x_data))
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Ensure we have tensors
assert torch.is_tensor(x_train), "x_train must be a tensor"
assert torch.is_tensor(y_train), "y_train must be a tensor"

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    train_outputs = model(x_train)
    train_loss = criterion(train_outputs, y_train)
    
    # Backward pass and optimization
    train_loss.backward()
    optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test)
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss.item(),
        "test_loss": test_loss.item()
    })
    writer.add_scalar('Loss/train', train_loss.item(), epoch)
    writer.add_scalar('Loss/test', test_loss.item(), epoch)

    if epoch % config.log_interval == 0:
        print(f'Epoch [{epoch}/{config.epochs}], '
              f'Train Loss: {train_loss.item():.4f}, '
              f'Test Loss: {test_loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), config.model_save_path)

# Close the writer
writer.close()
wandb.finish()

