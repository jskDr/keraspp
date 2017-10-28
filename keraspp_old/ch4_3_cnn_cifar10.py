"""
CH 9.1 Applications/Image Augmentation
"""
from keras import datasets
import keras
assert keras.backend.image_data_format() == 'channels_last'

from keraspp import aicnn


class Machine(aicnn.Machine):
    def __init__(self):
        (X, y), (x_test, y_test) = datasets.cifar10.load_data()
        super().__init__(X, y, nb_classes=10)


def main():
    m = Machine()
    m.run()

if __name__ == '__main__':
    main()