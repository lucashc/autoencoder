from urllib.request import urlopen
from zipfile import ZipFile
import os
from io import BytesIO
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import torch
from shutil import rmtree


BASE_URL = "http://www.viewfinderpanoramas.org/dem1/"

SAMPLES = 3601

# Data tiles of the Alps
TILES = ["n47e006",
         "n47e007",
         "n47e008",
         "n47e009",
         "n47e010",
         "n47e011",
         "n47e012",
         "n47e013",
         "n47e014",
         "n47e015",
         "n46e005",
         "n46e006",
         "n46e007",
         "n46e008",
         "n46e009",
         "n46e010",
         "n46e011",
         "n46e012",
         "n46e013",
         "n46e014",
         "n46e015",
         "n45e005",
         "n45e006",
         "n45e007",
         "n45e008",
         "n45e009",
         "n45e010",
         "n45e011",
         "n44e005",
         "n44e006",
         "n44e007",
         "n43e005",
         "n43e006",
         "n43e007"]


class Alps(Dataset):
    def __init__(self, directory, download=False, resolution=512, in_memory=False, transform=None, land_percentage=0.5):
        self.data_dir = directory
        self.tiles = os.path.join(directory, 'tiles/')
        self.npys = os.path.join(directory, 'npys/')
        self.resolution = resolution
        self.land_percentage = land_percentage
        if not os.path.isfile(os.path.join(self.data_dir, 'data.lock')):
            if download:
                os.makedirs(self.data_dir, exist_ok=True)
                os.makedirs(self.tiles, exist_ok=True)
                os.makedirs(self.npys, exist_ok=True)
                open(os.path.join(self.data_dir, "data.lock"), 'a').close()
                self.download_files()
                self.process()
                rmtree(self.tiles)
            else:
                raise FileNotFoundError("No lock file found")
        self.filelist = sorted(os.listdir(self.npys), key=lambda f: int(f[:-4]))
        self.samples_per_file = (SAMPLES // self.resolution)**2
        self.samples_row_column = SAMPLES // self.resolution
        self.length = self.samples_per_file * len(self.filelist)
        self.indices = list(range(self.length)) # All indices

        # Optional in memory storage
        self.in_memory = in_memory
        if self.in_memory:
            self.data = []
            self.load()
        # Remove corrupt indices
        self.mask_corrupted_indices()
        self.transform = transform

    def download_files(self):
        print("Downloading and extracting files...")
        for tile in tqdm(TILES):
            url = BASE_URL + tile.upper() + ".zip"
            resp = urlopen(url)
            zipfile = ZipFile(BytesIO(resp.read()))
            zipfile.extractall(self.tiles)
        print("Download and extraction done")
    
    def load(self):
        print("Loading in memory")
        for file in tqdm(self.filelist):
            self.data.append(np.load(os.path.join(self.npys, file))['data'])
        print("Loaded")

    @staticmethod
    def unpack_file(filename):
        with open(filename, 'rb') as f:
            elevations = np.fromfile(f, np.dtype('>i2'), SAMPLES*SAMPLES)\
                .reshape((SAMPLES, SAMPLES))
        return elevations
    
    @staticmethod
    def check_corruption(data, percentage=0.5):
        # Corruption by definition if there are values smaller than 0, and otherwise by percentage of land area.
	    return np.count_nonzero(data) < data.size*percentage or (data < 0).any()\
    
    def mask_corrupted_indices(self):
        removals = 0
        print("Pruning corrupted sectors")
        for index in tqdm(self.indices.copy()):
            file_index, _, y, x = self.calculate_file_tile(index)
            if self.in_memory:
                sample = self.data[file_index][y[0]:y[1], x[0]:x[1]]
            else:
                sample = np.load(os.path.join(self.npys, self.filelist[file_index]))['data'][y[0]:y[1], x[0]:x[1]]
            if self.check_corruption(sample, self.land_percentage):
                self.indices.remove(index)
                removals += 1
                tqdm.write(f"Corrupt sector at index {index}")
        self.length -= removals
        print(f"In total {removals} indices pruned")
    
    def process(self):
        print("Unpacking binary files...")
        for index, file in tqdm(enumerate(os.listdir(self.tiles))):
            data = self.unpack_file(os.path.join(self.tiles, file)).astype(np.float32)
            np.savez_compressed(os.path.join(self.npys, f"{index}.npz"), data=data)
        print("Unpacking done")
    
    def __len__(self):
        return self.length
    
    def calculate_file_tile(self, index):
        file_index = index // self.samples_per_file
        tile_index = index - file_index * self.samples_per_file
        row = tile_index // self.samples_row_column
        column = tile_index - self.samples_row_column * row
        y = (self.resolution * row, self.resolution*(row+1))
        x = (self.resolution * column, self.resolution*(column+1))
        return file_index, tile_index, y, x
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            index = idx.tolist()
        else:
            index = idx
        if index >= self.length or index < 0:
            raise IndexError("Out of bounds")
        # Only get file from allowed indices
        index = self.indices[index]
        file_index, _, y, x = self.calculate_file_tile(index)
        if self.in_memory:
            sample = self.data[file_index][y[0]:y[1], x[0]:x[1]]
        else:
            sample = np.load(os.path.join(self.npys, self.filelist[file_index]))['data'][y[0]:y[1], x[0]:x[1]]
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == "__main__":
    # Check if it works
    import time
    directory = str(time.time())
    x = Alps(directory, download=True, in_memory=True)
    for i in x:
        assert i.shape, (512, 512)
