# Towards the analysis of coral skeletal density-banding using Deep Learning

An implementation of a 2D U-Net architecture (initially based off of [zhixuhao's implementation](https://github.com/zhixuhao/unet)).

## Prerequisits

- `tensorflow<2`
- `keras==2.2.4`

These can be installed by running `pip install -r requirements.txt`.

If you plan on using a GPU to train, the `tensorflow-gpu` corresponding to the `tensorflow` version used is also required.

## Usage

### Training

```
$ python train.py --help
```

### Testing

```
$ python test.py --help
```

### Assessing the accuracy achieved

```
$ python accuracy.py --help
```

### Predicting an entire image

```
$ python predict.py --help
```