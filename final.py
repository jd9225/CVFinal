import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


#Train the model on imagenet with vgg16
# = VGG16(weights='imagenet', include_top=True)
#model.save("vgg16.h5")

#Load the saved model
model = tf.keras.models.load_model("D:\CSCI631-FoundCV\Final Project\FinalProject\\vgg16.h5")
model.layers.pop()
model.layers.pop()
print("vgg")
model.summary()

images = ['D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000103.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000108.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000113.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000118.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000123.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000128.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000133.jpg']
testimages = ['D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000304.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000309.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000314.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000319.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000324.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000329.jpg',
          'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000334.jpg']

predictimages = ['D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\000461.jpg',
                 'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\002439.jpg',
                 'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\003950.jpg',
                 'D:\CSCI631-FoundCV\Final Project\FinalProject\yolov3\\train\\004647.jpg']

def preprocessing(files):
    a = []
    for img_path in files:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x)
        a.append(features[np.newaxis, ...])

        f = model.layers[-2].output
        f = tf.squeeze(f, axis=0)

    inputarr = np.vstack(a)
    print(inputarr.shape)
    return inputarr


x_train = preprocessing(images)
x_test = preprocessing(testimages)
x_predict = preprocessing(predictimages)

#split into train and test
y_train = np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0]]) #y[:i]
y_test = np.array(([0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[1,0])) #y[i:]
y_predict = np.array(([0,1],[0,1],[0,1],[0,1])) #y[i:]

# #build model
# seq = Sequential()
# seq.add(LSTM(128,input_shape=(1,1000),return_sequences=True))
# seq.add(Dense(128, activation='relu'))
# seq.add(BatchNormalization())
#
# seq.add(LSTM(64,return_sequences=True))
# seq.add(Dense(64, activation='relu'))
# seq.add(BatchNormalization())
#
# seq.add(LSTM(32,return_sequences=True))
# seq.add(Dense(32, activation='relu'))
#
# seq.add(Flatten())
# seq.add(Dense(2, activation='softmax'))
#

#
# seq.summary()
# model = seq
#model.save("seq.h5")
model = tf.keras.models.load_model("D:\CSCI631-FoundCV\Final Project\FinalProject\\yolov3\\seq.h5")
print("seq")
model.summary()
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mae', 'acc'])
history = model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))

# predict accidents test images
pred_y = model.predict_classes(x_predict, verbose=True)

print(pred_y)

print(history)

# show graphs for accuarcy and loss
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


# confusion matrix
y_p = [0, 0, 0, 0]
y_predicted = model.predict_classes(x_predict)
print(y_predicted)
#y_pred = np.argmax(y_predicted, axis=1)

target_classes = [ "not accident", "accident" ]

print(confusion_matrix(np.asarray(y_p),y_predicted), )

print(classification_report(np.asarray(y_p),y_predicted,target_names = target_classes ))