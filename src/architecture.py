import warnings
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merging import concatenate
from keras.layers import BatchNormalization

warnings.filterwarnings('ignore')


def create_callbacks(backup_model_path: str, csv_log_path: str) -> tuple[any,any,any, any]:
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

    checkpointer = ModelCheckpoint(backup_model_path,
                                   monitor="val_loss",
                                   mode="min",
                                   save_best_only=True,
                                   verbose=1)

    earlystopper = EarlyStopping(monitor='val_loss',
                                 min_delta=0,
                                 patience=4,
                                 verbose=1,
                                 restore_best_weights=True)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=3,
                                   verbose=1,
                                   min_delta=1e-4)

    history_logger = CSVLogger(csv_log_path, append=True)

    return checkpointer, earlystopper, lr_reducer, history_logger


def create_model(image_height: int, image_width: int, dropout_factor: float = 1., filter_factor: int = 1) -> Model:
    dropout_prob_l1 = dropout_factor * 0.1
    dropout_prob_l2 = dropout_factor * 0.2
    dropout_prob_l3 = dropout_factor * 0.3

    f = filter_factor

    inputs = Input(shape=(image_height, image_width, 3))
    s = Lambda(lambda x: x / 255) (inputs)

    conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_01") (inputs)
    conv1 = BatchNormalization() (conv1)
    conv1 = Dropout(dropout_prob_l1) (conv1)
    conv1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_02") (conv1)
    conv1 = BatchNormalization() (conv1)
    pooling1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = Conv2D(f*32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_03") (pooling1)
    conv2 = BatchNormalization() (conv2)
    conv2 = Dropout(dropout_prob_l1) (conv2)
    conv2 = Conv2D(f*32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_04") (conv2)
    conv2 = BatchNormalization() (conv2)
    pooling2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = Conv2D(f*64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_05") (pooling2)
    conv3 = BatchNormalization() (conv3)
    conv3 = Dropout(dropout_prob_l2) (conv3)
    conv3 = Conv2D(f*64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_06") (conv3)
    conv3 = BatchNormalization() (conv3)
    pooling3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(f*128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_07") (pooling3)
    conv4 = BatchNormalization() (conv4)
    conv4 = Dropout(dropout_prob_l2) (conv4)
    conv4 = Conv2D(f*128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_08") (conv4)
    conv4 = BatchNormalization() (conv4)
    pooling4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

    conv5 = Conv2D(f*256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_09") (pooling4)
    conv5 = BatchNormalization() (conv5)
    conv5 = Dropout(dropout_prob_l3) (conv5)
    conv5 = Conv2D(f*256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_010") (conv5)
    conv5 = BatchNormalization() (conv5)


    upsample6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
    upsample6 = concatenate([upsample6, conv4])
    conv6 = Conv2D(f*128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_011") (upsample6)
    conv6 = BatchNormalization() (conv6)
    conv6 = Dropout(dropout_prob_l2) (conv6)
    conv6 = Conv2D(f*128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_012") (conv6)
    conv6 = BatchNormalization() (conv6)

    upsample7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv6)
    upsample7 = concatenate([upsample7, conv3])
    conv7 = Conv2D(f*64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_013") (upsample7)
    conv7 = BatchNormalization() (conv7)
    conv7 = Dropout(dropout_prob_l2) (conv7)
    conv7 = Conv2D(f*64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_014") (conv7)
    conv7 = BatchNormalization() (conv7)

    upsample8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv7)
    upsample8 = concatenate([upsample8, conv2])
    conv8 = Conv2D(f*32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_015") (upsample8)
    conv8 = BatchNormalization() (conv8)
    conv8 = Dropout(dropout_prob_l1) (conv8)
    conv8 = Conv2D(f*32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_016") (conv8)
    conv8 = BatchNormalization() (conv8)

    upsample9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv8)
    upsample9 = concatenate([upsample9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_017") (upsample9)
    conv9 = BatchNormalization() (conv9)
    conv9 = Dropout(dropout_prob_l1) (conv9)
    conv9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', name="conv2d_018") (conv9)
    conv9 = BatchNormalization() (conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
