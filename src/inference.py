# Imports
import glob

import keras
import numpy as np

from keras.models import load_model
#from keras.preprocessing.image import load_img
from skimage import transform
from PIL import Image
# Global Variables
IMG_HEIGHT, IMG_WIDTH = 512, 512 # 256, 256
CHANNELS = 3
ORIG_HEIGHT, ORIG_WIDTH = 0, 0

# Gives a tensor of size (1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
def image_makeup(img_filepath):
    np_img = Image.open(img_filepath)
    global ORIG_HEIGHT, ORIG_WIDTH
    ORIG_HEIGHT, ORIG_WIDTH = np_img.size
    np_img = np.array(np_img).astype('float32') 
    np_img = transform.resize(np_img, (IMG_WIDTH, IMG_HEIGHT, CHANNELS))

    print(np_img.shape, np_img[123,145,1])
    Image.fromarray(np_img.astype('uint8')).show()
    np_img = np.expand_dims(np_img, axis=0)
    print(np_img.shape, np_img[0][123,145,1])
    return np_img

def clean_up_predictions(preds) -> list:
    threshold = 0.50
    preds = preds * 255 #preds[preds > threshold] = 255
    preds = preds.astype('uint8')
    Image.fromarray(preds[0].reshape((512,512))).show()
    imgs = []
    for i in range(len(preds)):
        image = np.squeeze(preds[i][:, :, 0])
        image = Image.fromarray(image)
        image = image.resize((ORIG_HEIGHT, ORIG_WIDTH))
        imgs.append(image)
    return imgs

def predict_from_path(model: keras.Model, img_path: str, threshold = 0.1) -> Image:
    img = np.array(Image.open(img_path).convert("RGB"))
    return predict(model, img, threshold)
    #preds = model.predict(image_makeup(img_path)[0])
    #imgs_list = clean_up_predictions(preds)
    #imgs_list[0].show()

def predict(model: keras.Model, img:np.ndarray, threshold=0.1) -> Image:
    normalized = img.reshape((1, IMG_WIDTH, IMG_HEIGHT, CHANNELS)) / 255

    preds = model.predict(normalized)

    preds[preds > threshold] = 255
    output = [Image.fromarray(pred.reshape((512, 512))) for pred in preds]
    if len(output) != 1:
        raise Exception("output should be 1 but is" + str(len(output)))
    return output[0]

def evaluate(model: keras.Model, img:np.ndarray) -> Image:
    normalized = img.reshape((1, IMG_WIDTH, IMG_HEIGHT, CHANNELS)) / 255

    metrics = model.evaluate(normalized, batch_size=4)

    return metrics

def evaluate_from_path(model: keras.Model, img_path: str) -> Image:
    img = np.array(Image.open(img_path).convert("RGB"))
    return evaluate(model, img)


def predict_all_from_path(model: keras.Model, img_glob: str, threshold = 0.1) -> list[Image]:
    files = glob.glob(img_glob)
    return [predict_from_path(model, file, threshold) for file in files]

def predict_overlay(model: keras.Model, img: np.ndarray, threshold = 0.1, color=(255,0,0, 128)) -> Image:
    pred = predict(model, img, threshold).convert("RGB")
    pred.putalpha(0)
    pixels = list(pred.getdata())
    for i, p in enumerate(pixels):
        if p[0] == 255:
            pixels[i] = color
    pred.putdata(pixels)
    input = Image.fromarray(img, 'RGB')
    input.paste(pred, mask=pred)
    return input

def predict_overlay_from_path(model: keras.Model, img_path: str, threshold = 0.1) -> Image:
    img = np.array(Image.open(img_path).convert("RGB"))
    return predict_overlay(model, img, threshold=threshold)

def predict_all_overlay(model: keras.Model, img_glob: str, threshold = 0.1) -> list[Image]:
    files = glob.glob(img_glob)
    return [predict_overlay(model, file, threshold) for file in files]