import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# 분류 DNN 모델 구현 ########################
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.metrics import categorical_accuracy, categorical_crossentropy

class DNN():
    def __init__(self, Nin, Nh_l, Nout):
        self.X_ph = tf.placeholder(tf.float32, shape=(None, Nin))
        self.L_ph = tf.placeholder(tf.float32, shape=(None, Nout))
        
        # Modeling
        H = Dense(Nh_l[0], activation='relu')(self.X_ph)
        H = Dropout(0.5)(H)
        H = Dense(Nh_l[1], activation='relu')(H) 
        H = Dropout(0.25)(H)
        self.Y_tf = Dense(Nout, activation='softmax')(H)
        
        # Operation
        self.Loss_tf = tf.reduce_mean(
            categorical_crossentropy(self.L_ph, self.Y_tf))
        self.Train_tf = tf.train.AdamOptimizer().minimize(self.Loss_tf)
        self.Acc_tf = categorical_accuracy(self.L_ph, self.Y_tf)
        self.Init_tf = tf.global_variables_initializer()

# 데이터 준비 ##############################
import numpy as np
from keras import datasets  # mnist
from keras.utils import np_utils  # to_categorical

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)


# 학습 효과 분석 ##############################
from keraspp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt

def run(model, data, sess, epochs, batch_size=100):
    # epochs = 2
    # batch_size = 100
    (X_train, Y_train), (X_test, Y_test) = data
    sess.run(model.Init_tf)
    with sess.as_default():
        N_tr = X_train.shape[0]
        for epoch in range(epochs):
            for b in range(N_tr // batch_size):
                X_tr_b = X_train[batch_size * (b-1):batch_size * b]
                Y_tr_b = Y_train[batch_size * (b-1):batch_size * b]

                model.Train_tf.run(feed_dict={model.X_ph: X_tr_b, model.L_ph: Y_tr_b, K.learning_phase(): 1})
            loss = sess.run(model.Loss_tf, feed_dict={model.X_ph: X_test, model.L_ph: Y_test, K.learning_phase(): 0})
            acc = model.Acc_tf.eval(feed_dict={model.X_ph: X_test, model.L_ph: Y_test, K.learning_phase(): 0})
            print("Epoch {0}: loss = {1:.3f}, acc = {2:.3f}".format(epoch, loss, np.mean(acc)))

# 분류 DNN 학습 및 테스팅 ####################
def main():
    Nin = 784
    Nh_l = [100, 50]
    number_of_class = 10
    Nout = number_of_class

    data = Data_func()
    model = DNN(Nin, Nh_l, Nout)

    run(model, data, sess, 10, 100)


if __name__ == '__main__': 
    main()