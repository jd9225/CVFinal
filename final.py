import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, merge, Convolution2D, \
    LSTM, ConvLSTM2D, Input, TimeDistributed, MaxPooling2D, UpSampling2D, \
    Activation, Layer
from keras import backend
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D
from skimage.transform import rescale, resize, downscale_local_mean
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2 as cv

#load images
images = []
for seqnum in ["000103", "000108", "000113", "000118", "000123", "000128", "000133",
               "000431", "000436", "000441", "000446", "000451", "000456", "000461"]:
    im = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\{}.jpg".format(seqnum))
    img_clr = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    img = downscale_local_mean(img_clr, (2,2))
    images.append(img[np.newaxis, ...])
x_train = np.vstack(images)

print(x_train.shape)

images=[]
for seqnum in ["000304", "000309", "000314", "000319", "000324", "000329", "000334",
               "000514", "000519", "000524", "000529", "000534", "000539", "000544"]:
    imt = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\{}.jpg".format(seqnum))
    imgt_clr = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
    imgt = downscale_local_mean(imgt_clr, (2,2))
    images.append(imgt[np.newaxis, ...])
x_test = np.vstack(images)

print(x_test.shape)

imX = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000334.jpg")
imgX = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX_ds = downscale_local_mean(imgX, (2,2))
test_X = imgX_ds

imX2 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000461.jpg")
imgX2 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX2_ds = downscale_local_mean(imgX2, (2,2))
test_X2 = imgX2_ds

imX3 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\004647.jpg")
imgX3 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX3_ds = downscale_local_mean(imgX3, (2,2))
test_X3 = imgX3_ds

imX4 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\allwhite.jpg")
imgX4 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX4_ds = downscale_local_mean(imgX4, (2,2))
test_X4 = imgX4_ds

imX5 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\003062.jpg")
imgX5 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX5_ds = downscale_local_mean(imgX5, (2,2))
test_X5 = imgX5_ds

imX6 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\003727.jpg")
imgX6 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX6_ds = downscale_local_mean(imgX6, (2,2))
test_X6 = imgX6_ds

imX7 = cv.imread("D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\004682.jpg")
imgX7 = cv.cvtColor(imt, cv.COLOR_BGR2GRAY)
imgX7_ds = downscale_local_mean(imgX7, (2,2))
test_X7 = imgX7_ds

#test_X_d3 = np.array([test_X, test_X2])
test_X_d5 = np.vstack([ test_X[np.newaxis,...],
                        test_X2[np.newaxis,...],
                        test_X3[np.newaxis,...],
                        test_X4[np.newaxis,...],
                        test_X5[np.newaxis,...],
                        test_X6[np.newaxis,...],
                        test_X7[np.newaxis,...]])

sizew = img.shape[1:][0]
sizeh = img.shape[0]
print(sizeh, sizew)

#split into train and test
#i = int(0.8 * num_imgs)
#x_train = img[np.newaxis, np.newaxis, ..., np.newaxis]#np.zeros((1, 1, sizeh, sizew, 1))
#x_test = imgt[np.newaxis, np.newaxis, ..., np.newaxis]#np.zeros((1, 1, sizeh, sizew, 1))
y_train = np.array(([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],
                    [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0])) #y[:i]
y_test = np.array(([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],
                   [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0])) #y[i:]
#test_imgs = imgs[i:]
#test_bboxes = bboxes[i:]
#print("xtrain", x_train)

#build model
seq = Sequential()
seq.add(LSTM(128,input_shape=(360, 640), batch_input_shape=(7, 360, 640),
                    return_sequences=True, stateful=True))
seq.add(Dense(128, activation='relu'))
seq.add(BatchNormalization())

seq.add(LSTM(64,return_sequences=True, stateful=True))
seq.add(Dense(64, activation='relu'))
seq.add(BatchNormalization())

# seq.add(Attention())

seq.add(LSTM(32,return_sequences=True, stateful=True))
seq.add(Dense(32, activation='relu'))
seq.add(BatchNormalization())

seq.add(LSTM(16,return_sequences=True, stateful=True))
seq.add(Dense(2, activation='softmax'))
seq.add(Flatten())
seq.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
seq.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mae', 'acc'])

# seq.add(BatchNormalization())
#
# seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', data_format='channels_last'))

seq.summary()
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

history = model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test),
          batch_size=7)

#vec = test_X[np.newaxis, :]

# predict accidents test images
pred_y = model.predict_classes(test_X_d5, verbose=True)

# predict time to accident
# predTTE_y = model.predict_TTE()

print(pred_y)

print(history)

acc_vals = history.history['acc']
epochs = range(1, len(acc_vals)+1)
plt.plot(epochs, acc_vals, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

loss_values = history.history['mae']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()