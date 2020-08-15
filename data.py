import glob
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from keras.preprocessing.image import ImageDataGenerator


def adjust_data(image, label):
    """
    Adjust an image label pair to contain values between 0 and 1. The label
    is thresholded so it can only contain 0s or 1s.

    Args:
        image: (arr) an N dimensional array.
        label: (arr) an N dimensional array.
    Returns:
        image: (arr) the adjusted N dimensional image array.
        label: (arr) the adjusted N dimensional label array
    """
    
    if np.max(image) > 1:
        image = image / 255
        label = label / 255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return image, label


def train_generator(batch_size, train_path, image_folder, label_folder, aug_dict,
                    image_color_mode="grayscale", label_color_mode="grayscale",
                    image_save_prefix="image", label_save_prefix="label",
                    save_to_dir=None, target_size=(256, 256)):
    """
    Create a generator that yields image label pairs. Two separate generators are
    created that use the same random seed to allow them to augment the images
    and corresponding labels with the same transformations. These generators
    are then zipped into a single generator. To save the augmented images and
    labels generated, set the save_to_dir to the path to a directory.

    Args:
        batch_size: (int) the size of the batches to generate.
        train_path: (str) the data directory.
        image_folder: (str) the name of the image folder.
        label_folder: (str) the name of the label folder.
        aug_dict: (dict) a dictionary containing the transformations allowed
            during augmentation.
        image_color_mode: (str) the color mode that the images are stored in.
        label_color_mode: (str) the color mode that the labels are stored in.
        image_save_prefix: (str) the prefix that the saved images will have.
        label_save_prefix: (str) the prefix that the saved labels will have.
        save_to_dir: (str | None) if not None it specifies the directory to save
            the augmented images and labels to.
        target_size: (int, int) the size to reshape the images to during augmentation.
    Yields:
        image: (arr) the augmented image tensor of shape (batch_size, y, x, channels).
        label: (arr) the augmented label tensor of shape (batch_size, y, x, channels).
    """

    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)

    seed = np.random.randint(0, 100)

    # The same seed argument is used when the image and label generators are
    # created to ensure that the same transformations are applied to both.
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )

    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed
    )

    # Zip the two generators into one.
    train_generator = zip(image_generator, label_generator)

    # Before yielding the image label pair, pass them through the
    # adjust_data() method defined above.
    for image, label in train_generator:
        image, label = adjust_data(image, label)
        yield image, label


def test_generator(test_path, num_image=30, target_size=(256, 256)):
    """
    Create a generator that yields single 2D images with the correct tensor
    shape to be passed into the network.

    Args:
        test_path: (str) the test data directory.
        num_image: (int) the number of images to test.
        target_size: (int, int) the size to reshape the images to during augmentation.
    Yields:
        image: (arr) a single image tensor of shape (1, y, x, 1).
    """

    # Find all .png images in the specified directory.
    image_names = sorted(glob.glob(os.path.join(test_path, "*.png")))

    # Only yield num_images images.
    for i in range(num_image):
        image = io.imread(image_names[i], as_gray=True)
        image = image / 255
        image = trans.resize(image, target_size)
        # Add a dimension for the batch size (1) and a dimension for the
        # number of channels which is also 1 since the images
        # are grayscale.
        image = np.reshape(image, (1,) + image.shape + (1,))
        yield image


def save_result(save_path, npyfile):
    """
    Save the predictions made by the network to the path specified.

    Args:
        save_path: (str) the directory to save the images to.
        npyfile: (obj) the npyfile containing the results.
    """

    for i, item in enumerate(npyfile):
        image = item[:, :, 0]
        io.imsave(os.path.join(save_path, f"{i}_predict.png"), img_as_ubyte(image))