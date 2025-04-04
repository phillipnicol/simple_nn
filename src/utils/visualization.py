def log_metrics(epoch, loss, accuracy):
    import wandb
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})

def visualize_training(losses, accuracies):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def setup_tensorboard(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
    return writer

def log_tensorboard(writer, epoch, loss, accuracy):
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)