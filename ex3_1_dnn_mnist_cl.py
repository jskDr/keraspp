# 기본 파라미터 설정 #########################
Nin = 784
Nh_l = [100, 50]
number_of_class = 10
Nout = number_of_class
 
# 분류 DNN 모델 구현 ########################
from keras import layers, models
 
class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))       
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', 
                         optimizer='adam', 
                         metrics=['accuracy'])
        
# 데이터 준비 ##############################
import numpy as np 
from keras import datasets              
from keras.utils import np_utils 
 
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
 
L, W, H = X_train.shape
X_train = X_train.reshape(-1, W * H)
X_test = X_test.reshape(-1, W * H)
 
X_train = X_train / 255.0
X_test = X_test / 255.0 
 
# 분류 DNN 학습 및 테스팅 ####################
model = DNN(Nin, Nh_l, Nout)
history = model.fit(X_train, y_train, epochs=5, batch_size=100, validation_split=0.2)
performace_test = model.evaluate(X_test, y_test, batch_size=100)
print('Test Loss and Accuracy ->', performace_test)
