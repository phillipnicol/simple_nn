import torch
import numpy as np

class Dataset:
    def __init__(self, num_points=1000):
        self.num_points = num_points
        
    def get_data(self):
        # Generate random x values between -5 and 5
        x = np.linspace(-5, 5, self.num_points)
        # Generate y values according to the function
        y = np.sin(3*x) + np.exp(np.cos(x) + 0.1)
        
        # Convert to PyTorch tensors and ensure correct shape
        x_tensor = torch.FloatTensor(x).reshape(-1, 1)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        return x_tensor, y_tensor