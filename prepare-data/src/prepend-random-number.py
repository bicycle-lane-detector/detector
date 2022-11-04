from tqdm import tqdm
import os
import random

seed = 1337

def splitSource(img_path, mask_path, mask_extension):
    files = next(os.walk(img_path))[2]
    print('Total number of files =', len(files))

    random.seed(seed)

    for n, file in enumerate(tqdm(files, total=len(files))):
        number = random.randint(100, 999)

        os.rename(img_path + file, img_path + str(number) + "_" + file)
        os.rename(mask_path + file[:-3] + mask_extension, mask_path + str(number) + "_" + file[:-3] + mask_extension)

mask_extension = "png"
img_path = "./Data/all-cities/_Train512/Images/samples/"
mask_path = "./Data/all-cities/_Train512/Masks/samples/"

splitSource(img_path, mask_path, mask_extension)