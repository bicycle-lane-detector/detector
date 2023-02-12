import albumentations as A 
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random
from typing import Literal, Tuple
import cv2

class Augmentor:
    def __init__(self, probability_of_color=0.3, probability_of_translation=0.7, seed=1337):
        random.seed(seed)

        prob = probability_of_color
        
        transforms = [
            A.RandomBrightnessContrast(p=prob),
            A.ColorJitter(p=prob),
            A.RGBShift(p=prob),
            A.ChannelShuffle(p=prob),
        ]
        self.transformer_color = A.Compose(transforms)

        prob_trans = probability_of_translation
        self.transformer_translation = A.Compose([
            A.ShiftScaleRotate(scale_limit=0, shift_limit=0.1, rotate_limit=90,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=prob_trans)
        ])

    def batch_augment_x(self, img_batch: np.ndarray):
        for i in range(0, img_batch.shape[0]):
            img_batch[i,...] = self.augment_x(img_batch[i,...].astype('uint8'))
        return img_batch

    def batch_augment_x_y(self, img_batch: np.ndarray, mask_batch: np.ndarray):
        for i in range(0, img_batch.shape[0]):
            print("ITERATION XY BEFORE", i)
            print(np.unique(img_batch[i, ...]), img_batch[i, ...].shape)
            print("Y", i)
            print(np.unique(mask_batch[i, ...]), mask_batch[i, ...].shape)
            img_batch[i,...], mask_batch[i,...] = self.augment_x_y(img_batch[i,...].astype('uint8'), mask_batch[i,...])
            print("ITERATION XY AFTER", i)
            print(np.unique(img_batch[i, ...]), img_batch[i, ...].shape)
            print("Y", i)
            print(np.unique(mask_batch[i, ...]), mask_batch[i, ...].shape)
        return img_batch, mask_batch

    def augment_x(self, img: np.ndarray) -> np.ndarray:
        return self.transformer_color(image=img)['image']

    def augment_x_y(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        res = self.transformer_translation(image=img, mask=mask)
        return res['image'], res['mask']

def _visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def _to_disk(buf, name):
    img = Image.fromarray(buf)
    img.save(f"{name}.png")
    

def example_augmentations(which: Literal['one_by_one', 'all'] = 'one_by_one'):
    sample_image = np.array(Image.open("Data/karlsruhe/_Test512/Images/samples/149_karlsruhe.png"))
    random.seed(1337)
    prob = 1
    if which == 'all':
        prob = 0.5
        random.seed(42)
    
    transforms = [
        A.RandomBrightness(p=prob, limit=(0.2,0.2)),
        A.RandomContrast(p=prob, limit=(0.3,0.3)),#, brightness_limit=0),
        A.ColorJitter(p=prob),
        A.RGBShift(p=prob, r_shift_limit=20, g_shift_limit=(30,30), b_shift_limit=(20,20)),
        A.ChannelShuffle(p=prob),
    ]
    
    if which == 'all':
        t = A.Compose(transforms)
        res = t(image=sample_image)['image']
        _to_disk(res, 'all')
        return
    
    for t in transforms:
        res = t(image=sample_image)['image']
        name = type(t).__name__
        _to_disk(res, name)

# a = AdvancedAugmentor()
# sample_image = np.array(Image.open("Data/karlsruhe/_Test512/Images/samples/149_karlsruhe.png"))
# for i in range(50):
#     _to_disk(a.augment(sample_image), f"_{i}")