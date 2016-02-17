# tensorflow liftbridge

Basic training and evaluation python routines based off of the CIFAR10 tensorflow image model.

## Requirements

You'll need the tensorflow package for python3, located [here](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html).
These routines have only been tested with tensorflow built from source, but the pip3 packages should work.
GPU support for the liftbrige training has not been added.

## Getting started

The liftbridge images need to be processed into a binary format for training.

The processimages.py script in the 'tools' directory will do this for you. See the
readme file in that directory for more info.

When successful, processimages.py will create two files, train.bin and eval.bin.

Zip eval.bin and train.bin into a .tar.gz file. Modify liftbridge/liftbridge.py with the path to this file.

Finally, run
	python3 liftbridge_train.py

