'''
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer




# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 4, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/robin/ai/ineuron/main/3_computer_vision/dogcat/imgclf/images',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory('/home/robin/ai/ineuron/main/3_computer_vision/dogcat/imgclf/images',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'sparse')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,    
                         validation_steps = 2000)

classifier.save("model1.h5")
print("Saved model to disk")
print(training_set.class_indices)
# Part 3 - Making new predictions

'''


import numpy as np
from keras.preprocessing import image
import keras
test_image = image.load_img('/home/robin/ai/apps/solarsys/test_images/saturn01.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
new_model = keras.models.load_model('model1.h5')
result = new_model.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'earth'
    print(prediction)
elif result[0][1] == 1:
    prediction = 'jupiter'
    print(prediction)
elif result[0][2] == 1:
    prediction = 'mars'
    print(prediction)
elif result[0][3] == 1:
    prediction = 'saturn'
    print(prediction)
