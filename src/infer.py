import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from u_net import SimpleUnet

def sample_image_with_steps(
    model,      
    save_path,
    img_size=(1, 64, 64),
    T_steps=1000,
    snapshot_steps=[1000, 750, 500, 250, 0],  
    beta_start=1e-4,
    beta_end=0.02,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    betas = torch.linspace(beta_start, beta_end, T_steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)
    
    @torch.no_grad()
    def p_sample(model, x_t, t, t_index):
        betat = betas[t].to(device)
        alpha_t = alphas[t].to(device)
        alpha_bar_t = alpha_bars[t].to(device)
        alpha_bar_t_1 = alpha_bars[t-1].to(device) if t > 0 else alpha_bars[0].to(device)
        eps_pred = model(x_t)
        mu_t = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred)
    
        if t_index > 0:
            z = torch.randn_like(x_t)
        else:
            z = 0
        sigma_t = torch.sqrt(betat)
        x_t_minus_1 = mu_t + sigma_t * z
        return x_t_minus_1
    
    x = torch.randn((1, *img_size), device=device)
    snapshots = {}
    if T_steps in snapshot_steps:
        snapshots[T_steps] = x.clone()
    
    print("Generating image and capturing steps...")
    for t_index in tqdm(reversed(range(T_steps)), desc="Sampling"):
        t = torch.tensor([t_index], device=device, dtype=torch.long)
        x = p_sample(model, x, t, t_index)
        
        if t_index in snapshot_steps:
            snapshot_x = x.clone()
            snapshot_x = snapshot_x.clamp(-1, 1)
            snapshot_x = (snapshot_x + 1) * 0.5 
            snapshots[t_index] = snapshot_x
    
    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(15, 3))
    
    for i, step in enumerate(sorted(snapshots.keys(), reverse=True)):
        img = snapshots[step].squeeze().cpu().numpy()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"t = {step}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    
    print(f"Diffusion process visualization saved to {save_path}")
    return save_path


if __name__ == "__main__":

    model = SimpleUnet()
    model.load_state_dict(torch.load("../models/model_10.pt", map_location=torch.device('cpu')))
    model.eval()
    
    sample_image_with_steps(
        model,
        "../examples/diffusion_steps.png",
        img_size=(1, 64, 64),
        T_steps=1000,
        snapshot_steps=[1000, 750, 500, 250, 0],
        beta_start=1e-4,
        beta_end=0.02,
        device=torch.device('cpu')
    )