from pathlib import Path
from PIL import Image
from tqdm import tqdm


def tif_to_png(directory):
    # Get a list of all the TIF files in the directory
    path = Path(directory)
    tif_files = list(path.glob('*.tif'))

    # Create a new subdirectory to store the PNG files
    png_directory = path / (path.name + '_png')
    png_directory.mkdir(exist_ok=True)

    # Loop through each TIF file, convert it to PNG, and save it in the new subdirectory
    for tif_file in tqdm(tif_files, desc="Converting TIF to PNG", unit="file"):
        with Image.open(tif_file) as img:
            png_file = png_directory / (tif_file.stem + '.png')
            img.save(png_file, 'PNG')

tif_to_png("./../../Data/wolfsburg/_Test512/Images/samples")
tif_to_png("./../../Data/wolfsburg/_Train512/Images/samples")