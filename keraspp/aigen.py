"""
Generator code for the image data augmentation
- MIT Licence
- Author: Sungjin Kim
"""
from keras.preprocessing.image import ImageDataGenerator
from . import aicnn


class Machine_Generator(aicnn.Machine):
    def __init__(self, X, y, nb_classes=2, steps_per_epoch=10, fig=True,
                 gen_param_dict=None):
        super().__init__(X, y, nb_classes=nb_classes, fig=fig)
        self.set_generator(steps_per_epoch=steps_per_epoch,
                           gen_param_dict=gen_param_dict)

    def set_generator(self, steps_per_epoch=10, gen_param_dict=None):
        if gen_param_dict is not None:
            self.generator = ImageDataGenerator(**gen_param_dict)
        else:
            self.generator = ImageDataGenerator()

        print(self.data.X_train.shape)

        self.generator.fit(self.data.X_train, seed=0)
        self.steps_per_epoch = steps_per_epoch

    def fit(self, epochs=10, batch_size=64, verbose=1):
        model = self.model
        data = self.data
        generator = self.generator
        steps_per_epoch = self.steps_per_epoch

        history = model.fit_generator(
            generator.flow(data.X_train, data.Y_train, batch_size=batch_size),
            epochs=epochs, steps_per_epoch=steps_per_epoch,
            validation_data=(data.X_test, data.Y_test))

        return history
