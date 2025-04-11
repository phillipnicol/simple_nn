# Configuration settings for the PyTorch simple neural network project

class Config:
    def __init__(self):
        self.epochs = 1000
        self.learning_rate = 0.0001
        self.log_interval = 100
        self.tensorboard_log_dir = "runs/experiment"  # Added missing attribute
        self.model_save_path = "model.pth"

    # Hyperparameters
    #LEARNING_RATE = 0.001
    #BATCH_SIZE = 32
    #EPOCHS = 100

    # Paths
    MODEL_SAVE_PATH = './models/'
    LOGGING_PATH = './logs/'

    # Logging settings
    USE_WANDB = True
    USE_TENSORBOARD = True