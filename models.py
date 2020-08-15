from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate


def unet2D(classes=2, size=256, ablated=False):
    """
    Define the 2D U-Net architecture. An ablated architecture can be
    defined if ablated == True. This implementation is based heavily off
    of zhixuhao's implementation:

    https://github.com/zhixuhao/unet/blob/master/model.py

    Args:
        classes: (int) the number of classes that the network can predict.
        size: (int) the x and y dimensions of the input images.
        ablated: (bool) whether or not an ablated architecture should be
            defined. The ablated architecture contains no pass-forward
            of information from blocks in the contracting path to
            blocks in the expanding path.
    Returns:
        model: (Keras Model) returns a Keras model containing the layers
            as attributes.
    """

    input_size = (size, size, 1)
    inputs = Input(input_size)

    # Contracting path
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expanding path
    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = Conv2D(512, 2, activation="relu", padding="same")(up6)
    if not ablated:
        up6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(up6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation="relu", padding="same")(up7)
    if not ablated:
        up7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(up7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation="relu", padding="same")(up8)
    if not ablated:
        up8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(up8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation="relu", padding="same")(up9)
    if not ablated:
        up9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(up9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)
    conv9 = Conv2D(classes, 3, activation="relu", padding="same")(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    return Model(inputs=inputs, outputs=conv10)