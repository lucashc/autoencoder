from urllib.request import urlopen
from zipfile import ZipFile
import os
from io import BytesIO

base_url = "http://www.viewfinderpanoramas.org/dem1/"

# Data tiles of the Alps
tiles = ["n47e006",
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

# Download directory
try:
    os.makedirs("./data/")
except: pass
# Download zips

print("Start downloading...")

for index, tile in enumerate(tiles):
    print(f"Downloading file {index+1} of {len(tiles)}")
    url = base_url + tile.upper() + ".zip"
    resp = urlopen(url)
    zipfile = ZipFile(BytesIO(resp.read()))
    # Extract
    print("Extracting...")
    zipfile.extract(zipfile.namelist()[0], "./data")
print("Done.")
