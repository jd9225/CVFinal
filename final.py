import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, merge, Convolution2D, \
    LSTM, ConvLSTM2D, Input, TimeDistributed, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D
from skimage.transform import rescale, resize, downscale_local_mean

from keras_extensions import (categorical_crossentropy_3d_w, softmax_3d, softmax_2d)

from keras.optimizers import SGD
import cv2 as cv
import numpy as np

#load images
images = []
for seqnum in ["000103", "000108", "000113", "000118", "000123", "000128", "000133"]:
    im = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\{}.jpg".format(seqnum))
    img_clr = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    img = downscale_local_mean(img_clr, (2,2))
    images.append(img[np.newaxis, ..., np.newaxis])
x_train = np.vstack(images)
x_train = x_train[np.newaxis, ...]

print(x_train.shape)

images=[]
for seqnum in ["000304", "000309", "000314", "000319", "000324", "000329", "000334"]:
    imt = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\{}.jpg".format(seqnum))
    imgt_clr = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
    imgt = downscale_local_mean(imgt_clr, (2,2))
    images.append(imgt[np.newaxis, ..., np.newaxis])
x_test = np.vstack(images)
x_test = x_test[np.newaxis, ...]

print(x_test.shape)

imX = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000334.jpg")
imgX = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX_ds = downscale_local_mean(imgX, (2,2))
test_X = imgX_ds


imX2 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000461.jpg")
imgX2 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX2_ds = downscale_local_mean(imgX2, (2,2))
test_X2 = imgX2_ds

#test_X_d3 = np.array([test_X, test_X2])
test_X_d3 = np.vstack([ test_X[np.newaxis,np.newaxis,...,np.newaxis], test_X2[np.newaxis,np.newaxis,...,np.newaxis]])

sizew = img.shape[1:][0]
sizeh = img.shape[0]
print(sizeh, sizew)

#split into train and test
#i = int(0.8 * num_imgs)
#x_train = img[np.newaxis, np.newaxis, ..., np.newaxis]#np.zeros((1, 1, sizeh, sizew, 1))
#x_test = imgt[np.newaxis, np.newaxis, ..., np.newaxis]#np.zeros((1, 1, sizeh, sizew, 1))
y_train = np.array(([1])) #y[:i]
y_test = np.array(([1])) #y[i:]
#test_imgs = imgs[i:]
#test_bboxes = bboxes[i:]
#print("xtrain", x_train)

#build model
seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(7, 360, 640, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(Flatten())
seq.add(Dense(1, activation='softmax'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['mae', 'acc'])

# seq.add(BatchNormalization())
#
# seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', data_format='channels_last'))

#model.compile(loss=categorical_crossentropy_3d_w(2, class_dim=-1), optimizer='adadelta')
model = seq

#train
# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
#
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=opt,
#     metrics=['accuracy'],
# )

model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))

#vec = test_X[np.newaxis, :]

# predict accidents test images
pred_y = model.predict(test_X_d3)

print(pred_y)
