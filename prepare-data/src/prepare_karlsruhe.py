import numpy as np

from tqdm import tqdm
import os
import cv2
import time
import math

from PIL import Image




def crop_and_save(images_path, masks_path, new_images_path, new_masks_path, img_width, img_height):
    """
    same as in build_dataset.py with minor adjustments
    """

    print("Building Dataset.")

    num_skipped = 0
    start_time = time.time()
    files = next(os.walk(images_path))[2]
    print('Total number of files =', len(files))

    for image_file in tqdm(files, total=len(files)):

        image_path = images_path + image_file
        image = cv2.imread(image_path)

        mask_path = masks_path + image_file[:-3] + "png"
        mask = cv2.imread(mask_path, 0)

        num_splits = math.floor((image.shape[0] * image.shape[1]) / (img_width * img_height))
        counter = 0
        print(image.shape[0], image.shape[1])
        for r in range(0, image.shape[0], img_height):
            for c in range(0, image.shape[1], img_width):
                counter += 1
                blank_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                blank_mask = np.zeros((img_height, img_width), dtype=np.uint8)

                new_image_path = new_images_path + str(counter) + '_' + image_file
                new_mask_path = new_masks_path + str(counter) + '_' + image_file[:-3] + "png"

                new_image = np.array(image[r:r + img_height, c:c + img_width, :])
                new_mask = np.array(mask[r:r + img_height, c:c + img_width])

                blank_image[:new_image.shape[0], :new_image.shape[1], :] += new_image
                blank_mask[:new_image.shape[0], :new_image.shape[1]] += new_mask

                blank_mask[blank_mask > 1] = 255

                if not np.any(blank_mask):
                    continue
                
                cv2.imwrite(new_image_path, blank_image)
                cv2.imwrite(new_mask_path, blank_mask)


    print("EXPORT COMPLETE: {} seconds.\nImages exported to {}\nMasks exported to{}".format(
        round((time.time() - start_time), 2), new_images_path, new_masks_path))
    print("\n{} Images were skipped.".format(num_skipped))

def train_test_split(images_path, masks_path, extended_images, extended_masks, test_split=0.3):
    """
    same as in build_dataset.py with minor adjustments
    """

    image_filenames = [filename for filename in os.walk(images_path)][0][2]
    test_set_size = int(test_split * len(image_filenames))
    np.random.shuffle(image_filenames)

    root_path = os.path.dirname(os.path.dirname(images_path)) + "/"
    train_dir = root_path + extended_images
    test_dir = root_path + extended_masks

    if not os.path.exists(train_dir):
        print("CREATING:", train_dir)
        os.makedirs(train_dir + "Images/samples/")
        os.makedirs(train_dir + "Masks/samples/")

    if not os.path.exists(test_dir):
        print("CREATING:", test_dir)
        os.makedirs(test_dir + "Images/samples/")
        os.makedirs(test_dir + "Masks/samples/")

    train_image_dir = train_dir + "Images/samples/"
    train_mask_dir = train_dir + "Masks/samples/"
    test_image_dir = test_dir + "Images/samples/"
    test_mask_dir = test_dir + "Masks/samples/"

    for n, filename in enumerate(image_filenames):
        if n < test_set_size:
            os.rename(images_path + filename, test_image_dir + filename)
            os.rename(masks_path + filename[:-3] + "png", test_mask_dir + filename[:-3] + "png")
        else:
            os.rename(images_path + filename, train_image_dir + filename)
            os.rename(masks_path + filename[:-3] + "png", train_mask_dir + filename[:-3] + "png")

    print("Train-Test-Split COMPLETED.\nNUMBER OF IMAGES IN TRAIN SET:{}\nNUMBER OF IMAGES IN TEST SET: {}".format(
        len(image_filenames) - test_set_size, test_set_size))
    print("\nTrain Directory:", train_dir)
    print("Test Directory:", test_dir)

if __name__ == "__main__":
    root_data_path = "./Data/karlsruhe-small/"
    img_width = img_height = 256 * 2 #* 2
    num_channels = 3

    # Path Information
    images_path = root_data_path + "Images/"
    masks_path = root_data_path + "Masks/"
    new_images_path = root_data_path + "Images512/"
    new_masks_path = root_data_path + "Masks512/"
    extended_images = "_Train512/"
    extended_masks = "_Test512/"

    for path in [new_images_path, new_masks_path]:
        if not os.path.exists(path):
            os.mkdir(path)
            print("DIRECTORY CREATED: {}".format(path))
        else:
            print("DIRECTORY ALREADY EXISTS: {}".format(path))

    #splitSource(root_data_path + "source/", root_data_path + "Images/", root_data_path + "Masks/")
    crop_and_save(images_path, masks_path, new_images_path, new_masks_path, img_width, img_height)
    train_test_split(new_images_path, new_masks_path, extended_images, extended_masks, 1)
