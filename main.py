import torch
import os

from src.utils    import get_device
from src.dataset  import get_dataloader
from src.model    import get_model
from src.train    import train_model
from src.sample   import generate_images

def main():

    print("\n" + "="*40)
    print("  DIFFUSION MODEL FROM SCRATCH")
    print("="*40 + "\n")

    # Step 1 - Setup
    device     = get_device()
    dataloader = get_dataloader(batch_size=128)
    model      = get_model(device)

    # Step 2 - Train
    losses = train_model(
        model      = model,
        dataloader = dataloader,
        device     = device,
        n_epochs   = 3
    )

    # Step 3 - Save model
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/model.pth")
    print("Model saved -> outputs/model.pth")

    # Step 4 - Generate images
    generate_images(
        model    = model,
        device   = device,
        n_images = 64,
        n_steps  = 40
    )

    print("\n" + "="*40)
    print("ALL DONE! Check the outputs/ folder")
    print("="*40)
    print("  outputs/loss_curve.png      <- training loss graph")
    print("  outputs/generated_images.png <- generated digits")
    print("  outputs/model.pth            <- saved model")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()