import numpy as np
import os

file_list = os.listdir("./data")
try:
    os.makedirs("./npys")
except: pass

# Process files
SAMPLES = 3601


def unpack_file(filename):
    with open(filename, 'rb') as f:
        elevations = np.fromfile(f, np.dtype('>i2'), SAMPLES*SAMPLES)\
            .reshape((SAMPLES, SAMPLES))
    return elevations


# lat, lon = 45.909937, 5.866246
# lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
# lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))
# indices = (SAMPLES-1 - lat_row , lon_row)
# print(unpack_file("./data/N45E005.hgt")[indices[0], indices[1]].astype(int))

for index, file in enumerate(file_list):
    print(f"Unpacking file {index+1} of {len(file_list)}")
    data = unpack_file(os.path.join("./data/", file)).astype(int)
    np.save(os.path.join("./npys/", file[:-4] + ".npy"), data)
print("Done.")
