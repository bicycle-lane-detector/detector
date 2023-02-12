"""
Filename: get_data_generators.py

Function: Builds and returns data generators.

Author: Jerin Paul (https://github.com/Paulymorphous)
Website: https://www.livetheaiexperience.com/
"""
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from advanced_img_aug import Augmentor
from keras import layers
import tensorflow as tf
import numpy as np

class Generator:
    """ Data goes into keras and samples is the number of samples. samples per batch = size / batch_size"""
    data: tf.data.Dataset
    samples: int
    samplesPerBatch: any

    def __init__(self, data: tf.data.Dataset, samples: int, batch_size: int):
        self.data = data
        self.samples = samples
        self.samplesPerBatch = samples 


def getDataGenerators(augmentation_parameters, img_size, train_images_path=None, train_targets_path=None, test_images_path=None, test_targets_path=None, batch_size = 64, seed=42) -> list[Generator]:
    """
        Builds and returns ImageDataGenerators based on the paths that are sepcified.
        Each Generator will have data and size. Data goes int

        Please note:
        > Since this is not a multi-class classification problem and ImageDataGenerators require atleast one folder with images in it, we provide the path to the parent directory of the folder containting the images.
        > If you want only one type of data generator, then you need to provide the path only for that dataset. See Example for clarification.
        > A Validation generator is also returned with the Train generator.
        > You can modify the parameters for Image Augmentation in the Config File.

        Read more about the prerocessing methods here: https://keras.io/api/preprocessing/image/

        Parameters
        ----------
        >augmentation_parameters (Object of ConfigParser SectionProxy): Configurations from for Augmentation.
        >train_images_path (str): Path to the parent folder of the folder containting Train Images.
        >train_targets_path (str): Path to the parent folder of the folder containting Train Masks.
        >test_images_path (str): Path to the parent folder of the folder containting Test Images.
        >test_targets_path (str): Path to the parent folder of the folder containting Test Masks.
        >batch_size (int): Desired batch size for the datagenerators. Default: 64.
        >seed (int): this number seeds the Datagenerators. Default: 42, because its the answer to everything ;)

        Returns
        ----------
        > A list which can have some or all of the three types of Datagenerators - Train data generator, validation data generator, or/and Test data generator.

        Example
        ----------
        > If you want to test your model and only require the Test Datagenerator:
            test_generator = GetDataGenerators(test_images_path= <Path to test Images>,
                                            test_targets_path= <Path to test Maskss>,
                                            batch_size = 128
                                            )
    """


    generators = []

    normalize255 = lambda a : a/255
       
    if train_images_path and train_targets_path:

        use_aug = augmentation_parameters["use_aug"]
        val_split = 0.33 #float(augmentation_parameters["validation_split"])
    
        train_ds: tf.data.Dataset = image_dataset_from_directory(train_images_path,
                                                label_mode=None, 
                                                subset='training',
                                                image_size=img_size,
                                                batch_size=batch_size, 
                                                seed=seed,
                                                validation_split=val_split
                                                )
        
        val_ds: tf.data.Dataset = image_dataset_from_directory(train_images_path,
                                                label_mode=None, 
                                                subset='validation',
                                                image_size=img_size,
                                                batch_size=batch_size, 
                                                seed=seed,
                                                validation_split=val_split
                                                )
        
        train_target_ds: tf.data.Dataset = image_dataset_from_directory(train_targets_path,
                                                label_mode=None, 
                                                subset='training',
                                                image_size=img_size,
                                                batch_size=batch_size, 
                                                seed=seed,
                                                validation_split=val_split
                                                )
        
        val_target_ds: tf.data.Dataset = image_dataset_from_directory(train_targets_path,
                                                label_mode=None, 
                                                subset='validation',
                                                image_size=img_size,
                                                batch_size=batch_size, 
                                                seed=seed,
                                                validation_split=val_split
                                                )

        val_target_ds = val_target_ds.map(normalize255)
        train_target_ds = train_target_ds.map(normalize255)

        val = tf.data.Dataset.zip((val_ds, val_target_ds))
        train = tf.data.Dataset.zip((train_ds, train_target_ds))

        if use_aug:
            aug = Augmentor(seed=seed)
            def augment(x, y):

                x, y = aug.batch_augment_x_y(aug.batch_augment_x(x), y)
                return np.array([x, y])

            def wrap_numpy(x, y):
                res = tf.numpy_function(func=augment, inp=[x, y], Tout=tf.float32)
                return res[0], res[1]

            train = train.map(wrap_numpy)#, num_parallel_calls=tf.data.AUTOTUNE)

        val = val.prefetch(tf.data.AUTOTUNE)
        train = train.prefetch(tf.data.AUTOTUNE)

        generators.extend([Generator(train, train_ds.cardinality().numpy(), batch_size),
                           Generator(val, val_ds.cardinality().numpy(), batch_size)])

    
    if test_images_path and  test_targets_path:

        test_ds: tf.data.Dataset = image_dataset_from_directory(test_images_path,
                                                                      label_mode=None,
                                                                      image_size=img_size,
                                                                      batch_size=batch_size,
                                                                      shuffle=False,
                                                                      seed=seed,
                                                                      )

        test_target_ds: tf.data.Dataset = image_dataset_from_directory(test_targets_path,
                                                                      label_mode=None,
                                                                      image_size=img_size,
                                                                      batch_size=batch_size,
                                                                      shuffle=False,
                                                                      seed=seed,
                                                                      )

        test_target_ds = test_target_ds.map(normalize255)

        test_generator = tf.data.Dataset.zip((test_ds, test_target_ds)).prefetch(tf.data.AUTOTUNE)

        generators.append(Generator(test_generator, test_ds.cardinality().numpy(), batch_size))
    
    if generators:
        
        return generators
    
    else:
        print("Invalid Input for Data Paths. Plese Check and Retry.")
        return None