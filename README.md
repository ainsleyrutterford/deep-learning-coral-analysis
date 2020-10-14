# Towards the analysis of coral skeletal density-banding using Deep Learning

[Ainsley Rutterford](https://github.com/ainsleyrutterford), [Leonardo. Bertini](https://www.4d-reef.eu/avada-portfolio/leonardo-bertini/), [Erica J. Hendy](http://www.bris.ac.uk/earthsciences/people/erica-hendy/index.html), [Kenneth Johnson](https://www.nhm.ac.uk/our-science/departments-and-staff/staff-directory/kenneth-johnson.html), [Rebecca Summerfield](https://www.researchgate.net/profile/Rebecca_Summerfield), [Tilo Burghardt](http://people.cs.bris.ac.uk/~burghard/)

This repository contains accompanying code for [our note]() submitted to the Coral Reefs journal. X-ray micro-Computed Tomography (µCT) is increasingly used to record the skeletal growth banding of massive coral. However, the wealth of data generated is time-consuming to analyse and requires expert interpretation to estimate growth rates and colony age. We used a Keras-based Python implementation of the U-Net architecture [(Ronneberger et al. 2015)](https://arxiv.org/abs/1505.04597) as our backbone Convolutional Neural Network (CNN) to reproduce the expert identification of annual density banding. The CNN was trained with µCT images combined with manually-labelled ground truths to learn the topological features in different specimens of massive Porites sp. The CNN successfully predicted the position of low- and high-density boundaries in images not used in training.

<img src="https://github.com/ainsleyrutterford/deep-learning-coral-analysis/raw/master/coral.png">
<sup>An example of an X-ray µCT scan slice with the predicted high- and low-density boundaries superimposed.</sup>

<!-- Once published (hopefully!) how to cite section here. -->

## Repository overview

- [data/train](data/train), [data/test](data/test), and [data/val](data/val) contain the training, testing, and validation samples respectively.
- [data/splits](data/splits) contains various other train/test splits that can be used for cross-validation once the network is trained.
- [data.py](data.py) contains helper functions used to load and save data when training and testing.
- [models.py](models.py) contains a Keras implementation of the U-Net architecture.
- [train.py](train.py) is used to train the network on a given directory of training samples.
- [test.py](test.py) is used to test the network's performance on a given directory of testing samples.
- [predict.py](predict.py) is used to extract the density bands present in an entire 2D slice.
- [logs](logs) is an empty folder to which the TensorBoard log files will be saved.
- [test](test) is an empty folder to which the predictions output when testing the network will be saved.
- [utils/calcification.py](utils/calcification.py) is used to automatically calculate the densities, linear extension rates, and calcification rates of the slices stored in [utils/calcification](utils/calcification).
- [utils/calcification.ipynb](utils/calcification.ipynb) contains an interactive jupyter notebook which walks through the calculation of the density, linear extension rate, and calcification rate of a single area of a slice.
- [utils/sliding_window.py](utils/sliding_window.py) is used to generate a dataset of smaller "patches" that the network can train with.

## Prerequisits

- `tensorflow<2`
- `keras==2.2.4`

These can be installed by running `pip install -r requirements.txt`.

If you plan on using a GPU to train, the `tensorflow-gpu` corresponding to the `tensorflow` version used is also required. 

<sub>Note that this code was only tested with CUDA 10.1 and cuDNN 7.4. In order to use a newer version of CUDA or cuDNN, tensorflow may need to be updated. [This page](https://www.tensorflow.org/install/source#tested_build_configurations) contains a list of the recommended CUDA and cuDNN versions for each tensorflow version (windows users refer to [this page](https://www.tensorflow.org/install/source_windows#tested_build_configurations) instead). In order to check if a GPU is being used while training, the [train.py](train.py) script can be run with the `--verbose` flag.</sub>

## Generating a dataset

In order to generate a dataset of smaller patches, the [utils/sliding_window.py](utils/sliding_window.py) script can be used. To see what command line arguments are available, run

```
$ python utils/sliding_window.py --help
```

The script can be used for each slice one at a time. The following arguments must be supplied: the slice file name, the top left and bottom right coordinates of the confidently labelled area, the window size, and the stride. For example, with a top left coordinate `x1, y1`, a bottom right coordinate `x2, y2`, a window size of `20`, and a stride of `10`, run

```
$ python utils/sliding window.py slice.tif x1 y1 x2 y2 --size 20 --stride 10
```

## Training

The network can be trained using the [train.py](train.py) script. To see what command line arguments are available, run

```
$ python train.py --help
```

For example, to train the ablated 2D U-Net architecture with a learning rate of 0.0001 and a batch size of four, run

```
$ python train.py --ablated --lr 0.0001 --batch 4
```

Each sample that the network is trained is augmented. The augmentation parameters are defined in [train.py](train.py) and can be tuned if necessary.

## Testing

The [test.py](test.py) script can be used to test a trained network's performance on a dataset. To see what command line arguments are available, run

```
$ python test.py --help
```

For example, to test the network's performance on the [data/test](data/test) dataset, run

```
$ python test.py --dir data/test/image --tests 56
```

The resulting predictions are saved in the [test](test) directory.

## Estimating the calcification rate

In order to estimate the calcification rate of a given slice, the boundaries present in the slice must first be calculated using the [predict.py](predict.py) script. To see what command line arguments are available, run

```
$ python predict.py --help.
```

To predict the boundaries present in a slice named slice.png for example, one would run:

```
$ python predict.py --image slice.png
```

The image containing the skeletonized boundary positions will be saved in a file called out.png. Next, the [utils/calcification.py](utils/calcification.py) script can be used. The script will automatically estimate the density, linear extension rate, and calcification rate of the slice, and the final estimates will be printed by the last cell.

Coordinates and density calibration values of the slices we used are provided in the script. If you would like to estimate values for new slices, a `Slice` object must be defined with the following arguments: the slice image file name, the two sets of coordinates, and the density calibration values output by the CT machine.

## Acknowledgements 

The Keras U-Net implementation used was initially based off of [zhixuhao's implementation](https://github.com/zhixuhao/unet). This work was supported by NERC GW4+ Doctoral Training Partnership and is part of 4D-REEF, a Marie Sklodowska-Curie Innovative Training Network funded by European Union Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No. 813360.