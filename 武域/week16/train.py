import time
from models.unet_model import UNet
from utils.dataset import ISBI_Loader
import torch
from torch import optim
import torch.nn as nn

def calculate_accuracy(pred, label):
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred_binary = (pred >= 0.5).float()  # Convert probabilities to binary (0 or 1)
    correct = (pred_binary == label).sum().item()
    total = label.numel()
    return correct / total * 100

def train_net(net, device, data_path, epochs=100, batch_size=15, lr=1e-4, save_path="./models/best_model.pth", patience=3):

    # Load dataset
    dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize tracking for the best metrics
    best_accuracy = 0.0
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()
        net.train()  # Set model to training mode
        epoch_loss = 0.0
        total_accuracy = 0.0

        for batch_idx, (image, label) in enumerate(train_loader):
            # Move data to the appropriate device
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            pred = net(image)

            # Compute the loss
            loss = criterion(pred, label)
            epoch_loss += loss.item()

            # Calculate batch accuracy
            batch_accuracy = calculate_accuracy(pred, label)
            total_accuracy += batch_accuracy

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} Accuracy: {batch_accuracy:.2f}%")

        # Average metrics for the epoch
        epoch_accuracy = total_accuracy / len(train_loader)
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f} Average Accuracy: {epoch_accuracy:.2f}% Time: {epoch_time:.2f}s")

        # Check for improvement
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            epochs_no_improve = 0
            torch.save(net.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement in accuracy for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered. Training terminated.")
            break

    print("Training complete.")

if __name__ == '__main__':
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Initialize the U-Net model
    net = UNet(n_channels=1, n_classes=1)
    net.to(device)

    # Set the data path and train
    data_path = './data/train'
    train_net(net, device, data_path, epochs=100, batch_size=15, lr=1e-4)
