import torch 
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
class NoiseScheduler:
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_steps=200):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Create a linearly spaced array for beta values
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        
        # Calculate alphas as (1 - beta) for each timestep
        self.alphas = 1.0 - self.betas
        
        # Compute the cumulative product of alphas for easy computation
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute the square roots needed for the forward process
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)
        
    def add_noise(self, x_t, noise, t):
        if not isinstance(t, torch.LongTensor) and not isinstance(t, torch.cuda.LongTensor):
            t = t.long()
            
        batch_size = x_t.shape[0]
        
        # Reshape t to be a vector
        if isinstance(t, int) or (hasattr(t, 'shape') and len(t.shape) == 0):
            t = torch.ones(batch_size, device=x_t.device, dtype=torch.long) * t
    
        # Retrieve the precomputed square roots for the given timesteps
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cum_prod.to(x_t.device)[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t]

        # since x_1 is B x C x H x W, we need to unsqueeze the sqrt_alpha_cumprod_t and sqrt_one_minus_alpha_cumprod_t to B x C x H x W
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)

        # Apply the diffusion formula
        x_t_plus_1 = sqrt_alpha_cumprod_t * x_t + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t_plus_1
    
    def remove_noise(self, x_t, noise_pred, t):
        # Make sure t is a long tensor for indexing
        if not isinstance(t, torch.LongTensor) and not isinstance(t, torch.cuda.LongTensor):
            t = t.long()
            
        device = x_t.device
        
        # Estimate the original image x0 from the current image and noise prediction
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cum_prod.to(device)[t]
        sqrt_alpha = torch.sqrt(self.alpha_cum_prod.to(device)[t])
        
        # Reshape for broadcasting if necessary
        if len(sqrt_one_minus_alpha.shape) == 1 and len(x_t.shape) > 1:
            for _ in range(len(x_t.shape) - 1):
                sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
                sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        
        x0_estimated = (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        x0_estimated = torch.clamp(x0_estimated, -1.0, 1.0)

        beta_t = self.betas.to(device)[t]
        alpha_t = self.alphas.to(device)[t]
        
        if len(beta_t.shape) == 1 and len(x_t.shape) > 1:
            for _ in range(len(x_t.shape) - 1):
                beta_t = beta_t.unsqueeze(-1)
                alpha_t = alpha_t.unsqueeze(-1)
        
        mean = (x_t - (beta_t * noise_pred) / sqrt_one_minus_alpha) / torch.sqrt(alpha_t)
        
        if t.min() > 0:
            t_prev = t - 1
            variance = ((1.0 - self.alpha_cum_prod.to(device)[t_prev]) /
                       (1.0 - self.alpha_cum_prod.to(device)[t])) * beta_t
            
            # Reshape for broadcasting if necessary
            if len(variance.shape) == 1 and len(x_t.shape) > 1:
                for _ in range(len(x_t.shape) - 1):
                    variance = variance.unsqueeze(-1)
            
            standard_deviation = torch.sqrt(variance)
            z = torch.randn_like(x_t).to(device)
            x_t_minus_1 = mean + standard_deviation * z
            return x_t_minus_1, x0_estimated
        else:
            return mean, x0_estimated

if __name__ == "__main__":
    noise_scheduler = NoiseScheduler()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open("images/doggo.jpg")
    x_t = transform(image)

    B, C, H, W = 1, x_t.shape[0], x_t.shape[1], x_t.shape[2]
    print(x_t.shape)  
    print(torch.var(x_t))
    noise = torch.randn(B, C, H, W)
    t_100 = torch.tensor([100])
    x_t_plus_1 = noise_scheduler.add_noise(x_t, noise, t_100)
    print(x_t_plus_1.shape)
    print(torch.var(x_t_plus_1))

    t_150 = torch.tensor([150])
    x_t_plus_1_150 = noise_scheduler.add_noise(x_t, noise, t_150)
    print(x_t_plus_1_150.shape)
    print(torch.var(x_t_plus_1_150))

    #remove the B dimension for plotting
    x_t = x_t.squeeze(0)
    x_t_plus_1 = x_t_plus_1.squeeze(0)
    x_t_plus_1_150 = x_t_plus_1_150.squeeze(0)
    # plot the original image and the noisy image at t=100 and t=150
    plt.subplot(1, 3, 1)
    plt.imshow(x_t.permute(1, 2, 0))
    plt.subplot(1, 3, 2)
    plt.imshow(x_t_plus_1.permute(1, 2, 0))
    plt.subplot(1, 3, 3)
    plt.imshow(x_t_plus_1_150.permute(1, 2, 0))
    plt.show()

