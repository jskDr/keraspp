"""
Pretrained methods
- MIT Licence
- Sungjin Kim
"""
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16

from . import aicnn
from . import aigen


class CNN(aicnn.CNN):
    def __init__(model, input_shape, nb_classes,
                 n_dense=128, p_dropout=0.5, BN_flag=False,
                 PretrainedModel=VGG16):
        """
        If BN_flag is True, BN is used instaed of Dropout
        """
        model.in_shape = input_shape
        model.n_dense = n_dense
        model.p_dropout = p_dropout
        model.PretrainedModel = PretrainedModel
        model.BN_flag = BN_flag
        super().__init__(nb_classes)

    def build_model(model):
        nb_classes = model.nb_classes
        input_shape = model.in_shape
        PretrainedModel = model.PretrainedModel
        # print(nb_classes)

        # base_model = VGG16(weights='imagenet', include_top=False)

        base_model = PretrainedModel(
            weights='imagenet',
            include_top=False, input_shape=input_shape)

        x = base_model.input
        h = base_model.output
        z_cl = h  # Saving for cl output monitoring.

        h = model.topmodel(h)

        z_fl = h  # Saving for fl output monitoring.

        y = Dense(nb_classes, activation='softmax', name='preds')(h)
        # y = Dense(4, activation='softmax')(h)

        for layer in base_model.layers:
            layer.trainable = False

        model.cl_part = Model(x, z_cl)
        model.fl_part = Model(x, z_fl)

        model.x = x
        model.y = y

    def topmodel(model, h):
        '''
        Define topmodel
        if BN_Flag is True, BN is used instead of Dropout
        '''
        BN_flag = model.BN_flag

        n_dense = model.n_dense
        p_dropout = model.p_dropout

        h = GlobalAveragePooling2D()(h)
        h = Dense(n_dense, activation='relu')(h)
        if BN_flag:
            h = BatchNormalization()(h)
        else:
            h = Dropout(p_dropout)(h)
        return h


class DataSet(aicnn.DataSet):
    def __init__(self, X, y, nb_classes, n_channels=3, scaling=True,
                 test_size=0.2, random_state=0):
        self.n_channels = n_channels
        super().__init__(X, y, nb_classes, scaling=scaling,
                         test_size=test_size, random_state=random_state)

    def add_channels(self):
        n_channels = self.n_channels

        if n_channels == 1:
            super().add_channels()
        else:
            X = self.X
            if X.ndim < 4:  # if X.dim == 4, no need to add a channel rank.
                N, img_rows, img_cols = X.shape
                if K.image_dim_ordering() == 'th':
                    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
                    X = np.concatenate([X, X, X], axis=1)
                    input_shape = (n_channels, img_rows, img_cols)
                else:
                    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
                    X = np.concatenate([X, X, X], axis=3)
                    input_shape = (img_rows, img_cols, n_channels)
            else:
                if K.image_dim_ordering() == 'th':
                    N, Ch, img_rows, img_cols = X.shape
                    if Ch == 1:
                        X = np.concatenate([X, X, X], axis=1)
                    input_shape = (n_channels, img_rows, img_cols)
                else:
                    N, img_rows, img_cols, Ch = X.shape
                    if Ch == 1:
                        X = np.concatenate([X, X, X], axis=3)
                    input_shape = (img_rows, img_cols, n_channels)

            X = preprocess_input(X)
            self.X = X
            self.input_shape = input_shape


class Machine_Generator(aigen.Machine_Generator):
    """
    This Machine Generator is for pretrained approach.
    """
    def __init__(self, X, y, nb_classes=2, steps_per_epoch=10,
                 n_dense=128, p_dropout=0.5, BN_flag=False,
                 scaling=False,
                 PretrainedModel=VGG16, fig=True,
                 gen_param_dict=None):
        """
        scaling becomes False for DataSet
        """
        self.scaling = scaling
        self.n_dense = n_dense
        self.p_dropout = p_dropout
        self.BN_flag = BN_flag
        self.PretrainedModel = PretrainedModel

        # Machine class init
        super().__init__(X, y, nb_classes=nb_classes, steps_per_epoch=steps_per_epoch,
                       fig=fig, gen_param_dict=gen_param_dict)

    def set_data(self, X, y):
        nb_classes = self.nb_classes
        scaling = self.scaling
        self.data = DataSet(X, y, nb_classes, n_channels=3, scaling=scaling)

    def set_model(self):
        data = self.data
        nb_classes = self.nb_classes
        n_dense = self.n_dense
        p_dropout = self.p_dropout
        BN_flag = self.BN_flag
        PretrainedModel = self.PretrainedModel

        self.model = CNN(data.input_shape, nb_classes,
                         n_dense=n_dense, p_dropout=p_dropout, BN_flag=BN_flag,
                         PretrainedModel=PretrainedModel)

