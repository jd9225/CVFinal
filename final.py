import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, merge, Convolution2D, \
    LSTM, ConvLSTM2D, Input, TimeDistributed, MaxPooling2D, UpSampling2D
from keras_extensions import (categorical_crossentropy_3d_w, softmax_3d, softmax_2d)

from keras.optimizers import SGD
import cv2 as cv
import numpy as np

#load images
im = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000133.jpg")
img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#flatimg = img.flatten()

imt = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\004682.jpg")
imgt = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)

imX = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000334.jpg")
imgX = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
test_X = imgX

imX2 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000461.jpg")
imgX2 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
test_X2 = imgX2

test_X_d3 = np.array([test_X, test_X2])

sizew = img.shape[1:][0]
sizeh = img.shape[0]

#split into train and test
#i = int(0.8 * num_imgs)
x_train = np.zeros((1, sizeh, sizew))
x_test = np.zeros((1, sizeh, sizew))
y_train = np.zeros((1)) #y[:i]
y_test = np.zeros((1)) #y[i:]
#test_imgs = imgs[i:]
#test_bboxes = bboxes[i:]

#build model
c = 12
input_img = Input(x_train.shape[1:], name='input')
x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
c2 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
c3 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

x = TimeDistributed(UpSampling2D((2, 2)))(c3)
x = merge([c2, x], mode='concat')
x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

x = TimeDistributed(UpSampling2D((2, 2)))(x)
x = merge([c1, x], mode='concat')
# x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
x = TimeDistributed(UpSampling2D((2, 2)))(x)
x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
x = TimeDistributed(UpSampling2D((2, 2)))(x)

output = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', activation=softmax_2d(-1)), name='output')(x)

model = Model(input_img, output=[output])
#model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')


#train
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))

#vec = test_X[np.newaxis, :]

# predict accidents test images
pred_y = model.predict(test_X_d3)

print(pred_y)
