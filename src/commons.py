import keras
import datetime
import os
import numpy as np
import glob
from matplotlib import pyplot as plt

from advanced_img_aug import Augmentor

def save_model(model: keras.Model, model_path: str, csv_history_path_to_delete: str):

    print("Time of saving model:", datetime.datetime.now())

    model.save(model_path)
    os.remove(csv_history_path_to_delete)


def plot_history(history: any, model_path: str):
    loss = history.history['quality']
    val_loss = history.history['val_quality']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training Quality')
    plt.plot(epochs, val_loss, 'r', label='Validation Quality')
    plt.title('Training and validation quality')
    plt.xlabel('Epochs')
    plt.ylabel('Quality')
    plt.legend()
    plt.savefig(model_path[:-3] + "__quality.png")

    plt.show()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']

    plt.plot(epochs, acc, 'y', label='Training IoU')
    plt.plot(epochs, val_acc, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(model_path[:-3] + "__iou.png")
    plt.show()


def train_val_generator(IMAGE_WIDTH: int, IMAGE_HEIGHT: int, ROOT: str, TRAIN_DIR: str, BATCH_SIZE: int, seed: int, use_aug: bool) -> tuple[any, any] :
    import get_data_generators

    no_augmentation = {
        "use_aug": use_aug,
        "validation_split": 0.08
    }

    augmentation = {
        "use_aug": use_aug,
        "validation_split": 0.08
    }

    if not use_aug:
        augmentation = no_augmentation
    
    train, val = get_data_generators.getDataGenerators(augmentation, (IMAGE_WIDTH, IMAGE_HEIGHT),
                                                               ROOT + TRAIN_DIR + "Images", ROOT + TRAIN_DIR + "Masks",
                                                               batch_size=BATCH_SIZE, seed=seed)
    return train, val
