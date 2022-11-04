# Imports 
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
    print(np.unique(preds), preds[0].reshape((512,512)).shape)
    Image.fromarray(preds[0].reshape((512,512))).show()
    imgs = []
    for i in range(len(preds)):
        image = np.squeeze(preds[i][:, :, 0])
        image = Image.fromarray(image)
        image = image.resize((ORIG_HEIGHT, ORIG_WIDTH))
        imgs.append(image)
    return imgs

def predict(model, img_path) -> list:
    # model = load_model("./Models/road_mapper_final.h5", custom_objects = {
    #     "soft_dice_loss" : soft_dice_loss,
    #     "iou_coef" : iou_coef,
    #     "dice_coef_loss" : dice_coef_loss,
    #     "dice_loss" : dice_coef_loss,
    # })
    i = Image.open(img_path)
    i = i.convert("RGB")
    img = np.array(i)
    print(img.shape)
    preds = model.predict(img.reshape((1,IMG_WIDTH, IMG_HEIGHT, CHANNELS)) / 255)
    #preds = model.predict(image_makeup(img_path)[0])
    imgs_list = clean_up_predictions(preds)
    imgs_list[0].show()
    return imgs_list 




