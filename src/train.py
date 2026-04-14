import torch
from torch import nn
from src.utils import corrupt
import matplotlib.pyplot as plt
import os

def train_model(model, dataloader, device, n_epochs=3):
    """
    Train the diffusion model.
    Model learns to predict clean image from noisy image.
    """

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    print("\n" + "="*40)
    print("STARTING TRAINING")
    print("="*40)

    for epoch in range(n_epochs):
        
        epoch_losses = []
        
        for batch_idx, (x, _) in enumerate(dataloader):

            # Move data to device
            x = x.to(device)

            # Pick random noise amounts for each image
            noise_amount = torch.rand(x.shape[0]).to(device)

            # Corrupt the clean images
            noisy_x = corrupt(x, noise_amount)

            # Model predicts the clean image
            pred = model(noisy_x)

            # Calculate loss
            loss = loss_fn(pred, x)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} "
                      f"| Batch {batch_idx}/{len(dataloader)} "
                      f"| Loss: {loss.item():.5f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch+1} complete. Average Loss: {avg_loss:.5f}\n")

    print("="*40)
    print("TRAINING COMPLETE")
    print("="*40 + "\n")

    # Save loss curve
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.ylim(0, 0.1)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("outputs/loss_curve.png")
    plt.close()
    print("Loss curve saved -> outputs/loss_curve.png")

    return losses