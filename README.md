# BMVC2022_SVPG

This repository contains some basic (easy to run) codes for the experiments in the BMVC 2022 paper:

Biologically Plausible Variational Policy Gradient with Spiking Recurrent Winner-Take-All Networks

Citation at the bottom of this [link to paper](https://bmvc2022.mpi-inf.mpg.de/358/).

Note: An updated version of the experiments is available at [this repository](https://github.com/yzlc080733/SVPG2023).

## Prerequisites

System requirements: tested on Ubuntu 18.04, NVIDIA Geforce 3080, 32GB RAM, Anaconda3.

Part of installed packages: python(3.6), torch(1.8.2), snntorch(0.4), torchvision(0.9), scikit-image(0.17), opencv-python(4.5)

General running:

Simply run `python <code name>.py` to run the experiment with default parameters.

You may need to download the MNIST dataset. A preprocessing code is provided in `BMVC2022_SVPG/MNIST/MNIST_DATA/`.
Here are some of the parameters. See `parser` in the codes for more settings.

- `--cuda` sets the GPU to use;
  
- `--rep` sets the random seed;
  

Logs are stored in the `./log/` folder. Create it if it does not exist.

## Explanations

### MNIST

`MN_SVPG.py`, `MN_SVPG_shrink.py`:

Python codes respectively for the SVPG and SVPG-shrink methods.

`MNWTArate_nop.py`, `MNWTAuni_nop.py`, `MNWTAdexp_nop.py`:

Python codes for the comparison of the three implementations of SVPG, respectively rate coded with noise, spike coded with rectangle window, and spike coded with double exponential window.
