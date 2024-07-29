#train CNN model
from keras.models import Sequential#high level API for training deep learning model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()#stacking layers by linear fashion
model.add(Conv2D(32, (3, 3), activation=('relu'), input_shape=(20, 20, 3))) #adding first Convolutional layer #32 filters(3*3) #relu-all negative to zero #3color channels size 20*20
model.add(MaxPooling2D((2, 2)))#max pooling
model.add(Conv2D(64,(3,3), activation=('relu')))#2nd CN layer
model.add(MaxPooling2D(2,2))
model.add(Flatten()) #2D into 3D
model.add(Dense(128, activation=('relu'))) #1st dense
model.add(Dense(128, activation=('relu'))) #2nd dense
model.add(Dense(6, activation=('softmax'))) #output layer #probability decimal values
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compile #adam for recognition #error calculation in expected and predicted
model.fit(test_batches, epochs=17) #train CN #repeat training for 17times

#train data
train_path = '/Users/ashleyc/Deeplearning/fresh_and_rotton/dataset/train'
test_path = '/Users/ashleyc/Deeplearning/fresh_and_rotton/dataset/test'
BATCH_SIZE = 10
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rescale=1/255.,
    horizontal_flip=True,
    vertical_flip=True
).flow_from_directory(
    directory=train_path,
    target_size=(20, 20),
    classes=['freshapples', 'freshbananas', 'freshoranges', 'rottenapples', 'rottenbananas','rottenorganges'],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1/255.
).flow_from_directory(
    directory=test_path,
    target_size=(20, 20),
    classes=['freshapples', 'freshbananas', 'freshoranges', 'rottenapples', 'rottenbananas','rottenorganges'],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

#data manipulation
#imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#loading directories + data manipulation
train_path = '/Users/ashleyc/Deeplearning/fresh_and_rotton/dataset/train'
test_path = '/Users/ashleyc/Deeplearning/fresh_and_rotton/dataset/test'
BATCH_SIZE = 10
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
    rescale=1/255.,
    horizontal_flip=True,
    vertical_flip=True
).flow_from_directory(
    directory=train_path,
    target_size=(20, 20),
    classes=['freshapples', 'freshbananas', 'freshoranges', 'rottenapples', 'rottenbananas','rottenorganges'],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb'
)
test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1/255.
).flow_from_directory(
    directory=test_path,
    target_size=(20, 20),
    classes=['freshapples', 'freshbananas', 'freshoranges', 'rottenapples', 'rottenbananas','rottenorganges'],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)
#building the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation=('relu'), input_shape=(20, 20, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64,(3,3), activation=('relu')))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation=('relu')))
model.add(Dense(128, activation=('relu')))
model.add(Dense(6, activation=('softmax')))
#evaluating the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(test_batches, epochs=17)
