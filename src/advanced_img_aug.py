import albumentations as A 
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random
from typing import Literal

class AdvancedAugmentor:
    def __init__(self, probability_of_each=0.2):
        prob = probability_of_each
        
        transforms = [
            A.RandomBrightness(p=prob),
            A.RandomContrast(p=prob),
            A.ColorJitter(p=prob),
            A.RGBShift(p=prob),
            A.ChannelShuffle(p=prob),
        ]
        
        self.transformer = A.Compose(transforms)

    def augment(self, img: np.ndarray) -> np.ndarray:
        return self.transformer(image=img)['image']


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

