"""
CH 9.1 Applications/Image Augmentation
"""
from sklearn import model_selection
from keras import datasets
import keras
assert keras.backend.image_data_format() == 'channels_last'

from keraspp import aigen


class Machine(aigen.Machine_Generator):
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        _, X, _, y = model_selection.train_test_split(x_train, y_train, test_size=0.02)
        X = X.astype(float)

        gen_param_dict = {'rotation_range': 10}

        super().__init__(X, y, nb_classes=10, gen_param_dict=gen_param_dict)


def main():
    m = Machine()
    m.run()


if __name__ == '__main__':
    main()