################################
# 공통 패키지 불러오기
################################
from keras.datasets import mnist
import numpy as np
from PIL import Image
import math
import os

import keras.backend as K

K.set_image_data_format('channels_first')
print(K.image_data_format)

################################
# GAN 모델링
################################
from keras import models, layers, optimizers


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(1,2,3))


class GAN(models.Sequential):
    def __init__(self, input_dim=64):
        """
        self, self.generator, self.discriminator are all models
        """
        super().__init__()
        self.input_dim = input_dim

        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()
        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)
        
        self.compile_all()

    def compile_all(self):
        # Compiling stage
        d_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.generator.compile(loss=mean_squared_error, optimizer="SGD")
        self.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    def GENERATOR(self):
        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    def DISCRIMINATOR(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh',
                                input_shape=(1, 28, 28)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def get_z(self, ln):
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, (ln, input_dim))

    def train_both(self, x):
        ln = x.shape[0]
        # First trial for training discriminator
        z = self.get_z(ln)
        w = self.generator.predict(z, verbose=0)
        xw = np.concatenate((x, w))
        y2 = [1] * ln + [0] * ln
        d_loss = self.discriminator.train_on_batch(xw, y2)

        # Second trial for training generator
        z = self.get_z(ln)
        self.discriminator.trainable = False
        g_loss = self.train_on_batch(z, [1] * ln)
        self.discriminator.trainable = True

        return d_loss, g_loss


################################
# GAN 학습하기
################################
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
        j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


def get_x(X_train, index, BATCH_SIZE):
    return X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]


def save_images(generated_images, output_fold, epoch, index):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        output_fold + '/' +
        str(epoch) + "_" + str(index) + ".png")


def load_data():
    (X_train, y_train), (_, _) = mnist.load_data()

    return X_train[:10]

def train(args):
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_fold = args.output_fold
    input_dim = args.input_dim

    os.makedirs(output_fold, exist_ok=True)
    print('Output_fold is', output_fold)

    X_train = load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])

    gan = GAN(input_dim)

    d_loss_ll = []
    g_loss_ll = []
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            x = get_x(X_train, index, BATCH_SIZE)

            d_loss, g_loss = gan.train_both(x)

            d_loss_l.append(d_loss)
            g_loss_l.append(g_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            z = gan.get_z(x.shape[0])
            w = gan.generator.predict(z, verbose=0)
            save_images(w, output_fold, epoch, 0)

        d_loss_ll.append(d_loss_l)
        g_loss_ll.append(g_loss_l)

    gan.generator.save_weights(output_fold + '/' + 'generator', True)
    gan.discriminator.save_weights(output_fold + '/' + 'discriminator', True)

    np.savetxt(output_fold + '/' + 'd_loss', d_loss_ll)
    np.savetxt(output_fold + '/' + 'g_loss', g_loss_ll)


################################
# GAN 예제 실행하기
################################
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=2,
        help='Batch size for the networks')
    parser.add_argument('--epochs', type=int, default=1000,
        help='Epochs for the networks')
    parser.add_argument('--output_fold', type=str, default='GAN_OUT',
        help='Output fold to save the results')
    parser.add_argument('--input_dim', type=int, default=2,
        help='Input dimension for the generator.')

    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()