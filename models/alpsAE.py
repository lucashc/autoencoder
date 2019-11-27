import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.alps import Alps
import numpy as np

class AE(nn.Module):
    """
    AutoEncoder for the alps
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                # Input is (N, 1, 512, 512)
                nn.Conv2d(1, 16, 2, stride=2), # Shape (N, 16, 256, 256)
                nn.MaxPool2d(2, stride=2), # Shape (N, 16, 128, 128)
                nn.Conv2d(16, 8, 2, stride=2), # Shape (N, 8, 64, 64)
                nn.MaxPool2d(2, stride=2) # Shape (N, 8, 32, 32)
            )
        self.decoder = nn.Sequential(
                # Transpose convolution operator
                nn.ConvTranspose2d(8, 16, 2, stride=2), # Shape (N, 16, 64, 64)
                nn.ConvTranspose2d(16, 8, 2, stride=2), # Shape (N, 8, 128, 128)
                nn.ConvTranspose2d(8, 2, 2, stride=2), # Shape (N, 2, 256, 256)
                nn.ConvTranspose2d(2, 1, 2, stride=2) # Shape (N, 1, 512, 512)
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":

    # Settings
    batch_size = 128
    epochs = 20

    from torchvision import transforms # Transforms for the dataset
    from torch.utils.data import DataLoader # Class that handles loading of data from class that implements: 
        # __get_item__ and __len__, it also allows parallel loading if data, and implements batches
    from torch.autograd import Variable # Tensors of this type, keep history of gradient functions
    from tqdm import tqdm # Progressbars
    # Define model, metrics and optimizers
    model = AE()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Transform images
    img_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.astype(np.float32)/6000), # Normalize and make floats
        transforms.ToPILImage(), # Allows us to use other transforms
        transforms.RandomCrop((512, 512)), # Take random crops of size 512x512
        transforms.ToTensor(), # Converts PIL-images to torch tensors
    ])

    # Load data
    dataset = Alps('./alps', download=True, in_memory=True, resolution=1024, transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Run training
    for epoch in tqdm(range(epochs)):
        for data in dataloader:
            img = data
            img = Variable(img)
            # Make forward pass
            output = model(img)
            # Determine loss
            loss = criterion(output, img)
            # Backward
            # Reset gradients
            optimizer.zero_grad()
            # Do backpropagation
            loss.backward()
            # Update
            optimizer.step()
        tqdm.write(f"Epoch {epoch}, loss = {loss.item():.4f}")
    torch.save(model.state_dict(), "./conv_alps_autoencoder.pth")

