# 파일명: ex1-1.py
import keras
import numpy

x = numpy.array([0, 1, 2, 3, 4])
y = x * 2 + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x[:2], y[:2], epochs=1000, verbose=0)

print('목표 결과:', y[2:])
print('예측 결과:', model.predict(x[2:]).flatten())
