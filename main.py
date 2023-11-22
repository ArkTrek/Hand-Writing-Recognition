import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# load data
def load_data(path):
    images = []
    labels = []
    for i in range(10):
        for img in ImageDataGenerator(rescale=1./255).flow_from_directory(f'{path}/{i}', target_size=(32, 32), classes=[i]):
            images.append(img[0])
            labels.append(img[1])
    return np.array(images), np.array(labels)

# prepare data
def prepare_data(images, labels):
    images = images.astype('float32')
    labels = to_categorical(labels)
    return images, labels

# load data
images, labels = load_data('path/to/handwriting/dataset')

# prepare data
images, labels = prepare_data(images, labels)

# split data into training and testing sets
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# convert labels to categorical
labels_train = to_categorical(np.argmax(labels_train, axis=1))
labels_test = to_categorical(np.argmax(labels_test, axis=1))

# normalize images
images_train = images_train / 255
images_test = images_test / 255

# build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(images_train, labels_train, batch_size=128, epochs=10, validation_data=(images_test, labels_test))

# evaluate the model
score = model.evaluate(images_test, labels_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])