import argparse

# All possible arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=256, help="Size to reshape the images to when training")
parser.add_argument("--tests", type=int, default=56, help="The number of tests to carry out")
parser.add_argument("--weights", type=str, default="checkpoint-20.hdf5", help="Weights file to load")
parser.add_argument("--verbose", action="store_true", help="Show TensorFlow startup messages and warnings")
parser.add_argument("--ablated", action="store_true", help="Use ablated architecture")
parser.add_argument("--dir", type=str, default="data/test/image", help="Data directory")
args = parser.parse_args()

# If the --verbose argument is not supplied, suppress all of the TensorFlow startup messages.
if not args.verbose:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

import models
import data
from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

# Compile the model
model = models.unet2D(size=args.size, ablated=args.ablated)
model.compile(optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
model.load_weights(args.weights)

# Process the images in the test set and save the results to the test/ directory.
test_gen = data.test_generator(f"{args.dir}", num_image=args.tests, target_size=(args.size, args.size))
results = model.predict_generator(test_gen, args.tests, verbose=1)
# Use the custom save_result() method defined in data.py to save the results as
# greyscale .png images.
print("Saving results in ./test/")
data.save_result("test", results)
