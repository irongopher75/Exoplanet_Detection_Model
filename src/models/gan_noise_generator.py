"""
GAN-based noise generator for exoplanet detection data augmentation.
Generates synthetic noise samples to simulate instrument drift and sensor aging.
"""
import torch
import torch.nn as nn

class NoiseGAN(nn.Module):
    def __init__(self, input_dim=100, output_dim=1000):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def generate_noise(self, batch_size=32, input_dim=100):
        z = torch.randn(batch_size, input_dim)
        return self.generator(z)
    def discriminate(self, x):
        return self.discriminator(x)
