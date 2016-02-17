import io
import os
import random
import struct
from PIL import Image

IMAGE_SIZE = 128

def open_and_resize_image(imageFile):
    im = Image.open(imageFile)
    return im.resize([IMAGE_SIZE, IMAGE_SIZE])    


def process_images(outFile, baseDir, label, imageSet, indexRange):
    for i in indexRange:
        im = open_and_resize_image(baseDir + imageSet[i])
        outFile.write(label)
        imData = list(im.getdata(0)) + list(im.getdata(1)) + list(im.getdata(2))
        data = struct.pack('B' * len(imData), *imData)
        outFile.write(data)


# List and count files in each directory
upFiles = os.listdir('./up/')
downFiles = os.listdir('./down/')

minCount = min(len(upFiles), len(downFiles))

rUp = random.sample(upFiles, minCount)
rDn = random.sample(downFiles, minCount)

trainOut = io.open('train.bin', 'wb')
evalOut = io.open('eval.bin', 'wb')

halfCount = minCount / 2

process_images(trainOut, './up/', '1', rUp, range(0, halfCount))
process_images(trainOut, './down/', '0', rDn, range(0, halfCount))

process_images(evalOut, './up/', '1', rUp, range(halfCount, minCount))
process_images(evalOut, './down/', '0', rDn, range(halfCount, minCount))


trainOut.close()
evalOut.close()
