# processimages.py

processimages.py processes lift bridge images for tensorflow model training and evaluation.
The images are divided into two sets of equal size and randomly choosen for either training or evaluation.
Images are resized to 128x128. The binary format for each image consists of a 1-byte label, and RGB channels concatenated.

## Processing images

Put two folders in this directory, 'up' and 'down', containing lift bridge images in the up and down states, appropriately.

