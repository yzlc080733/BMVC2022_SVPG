# BMVC2022_SVPG

This repository contains some basic (easy to run) codes for the experiments in the BMVC paper:

No. 358

Biologically Plausible Variational Policy Gradient with Spiking Recurrent Winner-Take-All Networks

## Prerequisites
System requirements: tested on Ubuntu 18.04, NVIDIA Geforce 3080, 32GB RAM, Anaconda3.

Part of installed packages: python(3.6), torch(1.8.2), snntorch(0.4), torchvision(0.9), scikit-image(0.17), opencv-python(4.5)

General running:
	Simply run "python <code name>.py" to run the experiment with default parameters.
	Here are some of the parameters. See "parser" in the codes for more settings.
		"--cuda" sets the GPU to use;
		"--rep" sets the random seed;
	Logs are stored in the './log/' folder. Create it if it does not exist.
