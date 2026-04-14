import torch
import torchvision
import matplotlib.pyplot as plt
import os

def generate_images(model, device, n_images=64, n_steps=40):
    """
    Generate new images using trained model.
    Starts from pure noise and iteratively denoises.
    """

    print("\n" + "="*40)
    print("GENERATING IMAGES")
    print("="*40)

    model.eval()

    # Start from pure random noise
    x = torch.rand(n_images, 1, 28, 28).to(device)
    print(f"Starting from pure noise: {x.shape}")
    print(f"Running for {n_steps} steps...")

    for i in range(n_steps):
        with torch.no_grad():
            pred = model(x)

        # Move a little towards the prediction each step
        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor

        if i % 10 == 0:
            print(f"  Step {i}/{n_steps} done")

    print("Generation complete!")

    # Save the generated images
    os.makedirs("outputs", exist_ok=True)
    grid = torchvision.utils.make_grid(
        x.detach().cpu(), nrow=8
    )[0].clip(0, 1)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap='Greys')
    plt.title("Generated Digits")
    plt.axis('off')
    plt.savefig("outputs/generated_images.png", bbox_inches='tight')
    plt.close()

    print("Generated images saved -> outputs/generated_images.png")
    print("="*40 + "\n")

    return x