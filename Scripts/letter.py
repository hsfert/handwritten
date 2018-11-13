labels = []
files = []
import csv
import numpy as np
with open('../handwritten/letters.csv', 'rt',encoding="utf-8") as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
  count = 0
  for row in spamreader:
    if count > 0:
      labels.append(int(row[1])-1)
      files.append(row[2])
    count+=1
labels = np.array(labels)
from os.path import join
letter_paths_1 = [join('../handwritten/letters/', filename) for filename in files]
from IPython.display import display 
from PIL import Image
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
image_size = 32
def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
  imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
  img_array = np.array([img_to_array(img) for img in imgs])
  return preprocess_input(img_array)
letter_imgs_1 = read_and_prep_images(letter_paths_1)
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

img_rows, img_cols = 32, 32
num_classes = 33
out_y = keras.utils.to_categorical(labels, num_classes)
from tensorflow.python.keras.layers.advanced_activations import PReLU, LeakyReLU
from tensorflow.python.keras.layers import Activation, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

model = Sequential()
model.add(Conv2D(30, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 3)))
model.add(LeakyReLU(alpha=0.02))
    
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(196, (5, 5)))
model.add(LeakyReLU(alpha=0.02))
    
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(GlobalMaxPooling2D())
    
model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.02))
model.add(Dropout(0.5)) 
    
model.add(Dense(33))
model.add(Activation('softmax'))
    
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(letter_imgs_1, out_y,
          batch_size=128,
          epochs=200,
          validation_split = 0.2)
model.save_weights('../handwritten/model2.h5') 
model.load_weights('../handwritten/model2.h5')