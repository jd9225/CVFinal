import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
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
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='softmax'))


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
