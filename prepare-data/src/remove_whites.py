import numpy as np
import cv2

from tqdm import tqdm
import os
import math
import time

from PIL import Image

def redistributeTrainTest(ratio, trainImages, testImages):
    print("Redistributing images in a ratio of", ratio, "train to test")


    filesTrain = next(os.walk(trainImages))[2]
    lenTrain = len(filesTrain)
    print('Total number of train files =', lenTrain)
    filesTest = next(os.walk(testImages))[2]
    lenTest = len(filesTest)
    print('Total number of test files =', lenTest)

    filesToMove = (lenTrain + lenTest) * ratio - lenTrain

    small = ""
    large = ""
    files = []
    if filesToMove >= 0:
        files = filesTest
        small = testImages
        large = trainImages
    else:
        filesToMove *= -1
        files = filesTrain
        small = trainImages
        large = testImages
    np.random.shuffle(files)

    ctr = 0
    for image_file in tqdm(files, total=filesToMove):
        if ctr >= filesToMove:
            break
        ctr += 1
        pathImg = small + image_file
        pathMask = small + "../../Masks/samples/" + image_file
        newPathImg = large + image_file
        newPathMask = large + "../../Masks/samples/" + image_file

        os.rename(pathImg, newPathImg)
        os.rename(pathMask, newPathMask)

    print('Total number of train : test files has been rebalanced to ', len(next(os.walk(trainImages))[2]), ":", len(next(os.walk(testImages))[2]))

def copy_non_whites(images_path, masks_path, new_images_path, new_masks_path):
    print("Removing Whites.")

    num_skipped = 0
    start_time = time.time()
    files = next(os.walk(images_path))[2]
    print('Total number of files =', len(files))



    for image_file in tqdm(files, total=len(files)):

        image_path = images_path + image_file
        imgRaw = cv2.imread(image_path)
        image = np.array(imgRaw)

        counter = 0
        n_white = 0

        summedChanels = np.sum(image, axis=2)
        values, count = np.unique(summedChanels, return_counts=True)
        if (values[-1] == 765) and (count[-1] / (image.shape[0] * image.shape[1]) > 0.1):
            num_skipped += 1
            continue

        cv2.imwrite(new_images_path + image_file, imgRaw)
        imgMask = cv2.imread(masks_path + image_file)
        cv2.imwrite(new_masks_path + image_file, imgMask)

    print("EXPORT COMPLETE: {} seconds.\nImages exported to {}\nMasks exported to{}".format(
        round((time.time() - start_time), 2), new_images_path, new_masks_path))
    print("\n{} Images were skipped.".format(num_skipped))

if __name__ == "__main__":
    root = "./tiff/"

    copy_non_whites(root + "_Train512/Images/samples/", root + "_Train512/Masks/samples/",
                    root + "_Train512NoWhites/Images/samples/", root + "_Train512NoWhites/Masks/samples/")
    copy_non_whites(root + "_Test512/Images/samples/", root + "_Test512/Masks/samples/",
                    root + "_Test512NoWhites/Images/samples/", root + "_Test512NoWhites/Masks/samples/")

    #redistributeTrainTest(0.85, "./tiff/_TrainNoWhites/Images/samples/", "./tiff/_TestNoWhites/Images/samples/")