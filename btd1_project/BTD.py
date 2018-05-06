
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[2]:


classifier = Sequential()
# Input layer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# In[3]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Hidden layer 1
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Hidden layer 2

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Hidden layer 3
classifier.add(Flatten())

# Hidden layer 4
classifier.add(Dense(activation = 'relu',units=128))
classifier.add(Dense(activation = 'sigmoid',units=1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()


# In[4]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[5]:


import os 
os.getcwd()
os.chdir('/home/telraswa/Desktop/Swapnil/manju_project/Brain_tumor')
print(os.getcwd())


# In[6]:


training_set = train_datagen.flow_from_directory('/home/telraswa/Desktop/Swapnil/manju_project/Brain_tumor/train/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/home/telraswa/Desktop/Swapnil/manju_project/Brain_tumor/test/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[7]:


classifier.fit_generator(training_set, steps_per_epoch=None, epochs=100, verbose=1, callbacks=None, validation_data=test_set, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
     


# In[9]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/telraswa/Desktop/Swapnil/manju_project/TestImages/brain-tumors-fig2_large.jpg', target_size = (64, 64))
test_image


# In[10]:


test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
test_image


# In[11]:


result = classifier.predict(test_image)
result


# In[12]:


training_set.class_indices


# In[13]:


if result[0][0] == 0:
    prediction = 'Benign'
else:
    prediction = 'Malignent'
print("Detected tumor type is %s"%prediction)

