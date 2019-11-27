import torch
import torch.nn as nn

class Mons(nn.Module):

	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 32, 4, stride=4, padding=0), # (N x 32 x 128 x 128)
			nn.MaxPool2d(2, stide=2), # (N x 32 x 64 x 64)
			nn.Conv2d(32, 64, 4, stride=2, padding=0), # (N x 64 x 32 x 32)
			nn.MaxPool(2, stride=2), # (N x 64 x 16 x 16)
			nn.Conv2d(64, 128, 8, stride=8, padding=0), # (N x 128 x 2 x 2)
			nn.MaxPool(2, stride=2) # (N x 128 x 1 x 1)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 128, 4, stride=4),
			nn.ConvTranspose2d(128, 64, 4, stride=4),
			nn.ConvTranspose2d(64, 32, 4, stride=2),
			nn.ConvTranspose2d(32, 16, 4, stride=4),
			nn.ConvTranspose2d(16, 1, 4, stride=4)
		)

	def forward(self, x):
		return self.decoder(self.encoder(x))
