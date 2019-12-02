from torchvision import transforms
import torchvision.transforms.functional as F_trans
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch.nn import MSELoss
from tqdm import tqdm
import argparse
from alps import Alps
import pyaml
from sys import argv
import numpy as np
import importlib

args = argparse.ArgumentParser("Alps Model runner")
args.add_argument("model", help="Model to use", type=str)
args.add_argument("--load", help="Savefile to load")
args.add_argument("-b", "--batchsize", default=128, type=int)
args.add_argument("-e", "--epochs", default=20, type=int)
args.add_argument("--learningrate", default=1e-3, type=float)
args.add_argument("--weightdecay", default=1e-4, type=float)
args.add_argument("--save")

settings = args.parse_args()

batch_size = settings.batchsize
epochs = settings.epochs
save = settings.save
print("Using settings: ")
print(settings)

try:
    module = importlib.import_module('models.' + settings.model)
    AE = module.AE
except ImportError:
    print("Model not found")
    exit(-1)


# Define model, metrics and optimizers
if settings.load:
    model = AE()
    model.load_state_dict(torch.load(settings.load))
    model.eval()
else:
    model = AE()
criterion = MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=settings.learningrate, weight_decay=settings.weightdecay)

# Transform images
img_transform = transforms.Compose([
    transforms.Lambda(lambda x: x/704), # Normalize and make floats
    transforms.ToPILImage(), # Allows us to use other transforms
    transforms.RandomCrop((512, 512)), # Take random crops of size 512x512
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.Lambda(lambda x: F_trans.rotate(x, 90)),
        transforms.Lambda(lambda x: F_trans.rotate(x, -90)),
        transforms.Lambda(lambda x: F_trans.rotate(x, 180)),
        transforms.Lambda(lambda x: x)
    ]),
    transforms.ToTensor(), # Converts PIL-images to torch tensors
])

# Load data
dataset = Alps('../alps', download=True, in_memory=True, resolution=1024, transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Run training
try:
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
        if save:
            x = len(str(epochs))
            np.savez(f"{epoch:0{x}}", output=output.data.numpy(), input=img.data.numpy())
    if settings.save:
        torch.save(model.state_dict(), settings.save)
except KeyboardInterrupt:
    if settings.save:
        torch.save(model.state_dict(), settings.save)


