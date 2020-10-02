# Towards the analysis of coral skeletal density-banding using Deep Learning

[Ainsley Rutterford](), [Leonardo. Bertini](), [Erica J. Hendy](), [Kenneth Johnson](), [Rebecca Summerfield](), [Tilo Burghardt]()

<p align="center">
  <img src="https://github.com/ainsleyrutterford/deep-learning-coral-analysis/raw/master/coral.png">
</p>

The accompanying code for [our note]() submitted to the Coral Reefs journal. We used a Keras-based Python implementation of the U-Net architecture [(Ronneberger et al. 2015)](https://arxiv.org/abs/1505.04597) as our backbone Convolutional Neural Network (CNN).

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

## Prerequisits

- `tensorflow<2`
- `keras==2.2.4`

These can be installed by running `pip install -r requirements.txt`.

If you plan on using a GPU to train, the `tensorflow-gpu` corresponding to the `tensorflow` version used is also required. 

<sub>Note that this code was only tested with CUDA 10.1 and cuDNN 7.4. In order to use a newer version of CUDA or cuDNN, tensorflow may need to be updated. [This page](https://www.tensorflow.org/install/source#tested_build_configurations) contains a list of the recommended CUDA and cuDNN versions for each tensorflow version (windows users refer to [this page](https://www.tensorflow.org/install/source_windows#tested_build_configurations) instead). In order to check if a GPU is being used while training, the [train.py](train.py) script can be run with the `--verbose` flag.</sub>

## Generating a dataset

In order to generate a dataset of smaller "patches", the [utils/sliding_window.py](utils/sliding_window.py) script can be used. To see what command line arguments are available, run

```
$ python utils/sliding_window.py --help
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

## Testing

```
$ python test.py --help
```

## Predicting an entire image

```
$ python predict.py --help
```

## Acknowledgements 

The Keras U-Net implementation used was initially based off of [zhixuhao's implementation](https://github.com/zhixuhao/unet). This work was supported by NERC GW4+ Doctoral Training Partnership and is part of 4D-REEF, a Marie Sklodowska-Curie Innovative Training Network funded by European Union Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No. 813360.