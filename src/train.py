import torch
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from u_net import SimpleUnet
from noise_scheduler import NoiseScheduler
from infer import sample_image_with_steps
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "torchvision", "numpy", "tqdm", "matplotlib")
app = modal.App(name="diffusion-model")
output_volume = modal.Volume.from_name("diffusion-model-outputs")
data_volume = modal.Volume.from_name("data")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
IMAGE_SIZE = 64
SAVE_DIR = "diffusion-model-outputs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.function(image=image, volumes={"/diffusion-model-outputs": output_volume}, gpu="H100", timeout=3600)
def train():
    print(f"Using device: {DEVICE}")
    scheduler = NoiseScheduler(
        beta_start=BETA_START,
        beta_end=BETA_END,
        num_steps=NUM_TIMESTEPS
    )
   
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(
        root="/data",
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    model = SimpleUnet(in_channels=1, out_channels=1).to(DEVICE)
    model.train()
    
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    # Train loop
    for epoch in range(EPOCHS):
        losses = []
        curr_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} (loss: {curr_loss:.4f})"):
            # Get images
            images = batch[0].to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Sample random noise
            noise = torch.randn_like(images).to(DEVICE)
            
            # Sample random timesteps
            t = torch.randint(0, NUM_TIMESTEPS, (images.shape[0],)).to(DEVICE)
            
            # Add noise to images according to timestep
            noisy_images = scheduler.add_noise(images, noise, t)
            
            # Get noise prediction from model
            # Since we don't use time embedding, we're predicting noise without knowing t
            # This still works but might limit model capacity
            noise_pred = model(noisy_images)
            
            # Calculate loss and update weights
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            curr_loss = loss.item()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"/diffusion-model-outputs/model_{epoch}.pt")
            sample_image_with_steps(model, f"/diffusion-model-outputs/samples_{epoch}.png", num_samples=4, img_size=(1, IMAGE_SIZE, IMAGE_SIZE), T_steps=NUM_TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE)
            print(f"Saved model and samples for epoch {epoch+1}")
    

@app.local_entrypoint()
def main():
    train.remote()