import torch
import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    """
    Simple AutoEncoder, loosely inspired from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                # Convolution: Input channel of 1, 16 channels of outputs, 3x3 convolution matrix
                # Per step, go 3 units (stride), pad with zeros on edges
                # Inputs are always of shape (N, C, H, W), N := batch size, C := Channels in input
                # H := Height of input image, W := Width of image
                nn.Conv2d(1, 16, 3, stride=3, padding=1), # Shape (N, 16, ...)
                # Max pooling: Take a 2x2 window (kernel), step with 2 units (stride)
                nn.MaxPool2d(2, stride=2), # Shape (N, 16, ...)
                nn.Conv2d(16, 8, 3, stride=2, padding=1), # Shape (N, 8, ...)
                nn.MaxPool2d(2, stride=1) # Shape (N, 8, ...)
            )
        self.decoder = nn.Sequential(
                # Transpose convolution operator
                nn.ConvTranspose2d(8, 16, 3, stride=2), # Shape (N, 16, ...)
                nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), # Shape (N, 8, ...)
                nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1) # Shape (N, 1, ...)
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    from torchvision.datasets import MNIST # Dataset
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
        transforms.ToTensor(), # Converts possible NumPy-arrays to torch tensors
        # Normalize with mean 0.5 and std 0.5 for each channel
        # input[channe] = (input[channel] - mean[channel]) / std[channel]
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load data
    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Run training
    for epoch in tqdm(range(100)):
        for data in dataloader:
            img, _ = data
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
    torch.save(model.state_dict(), "./conv_autoencoder.pth")


