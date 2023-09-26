#!/usr/bin/env python
# coding: utf-8

# Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

# In[1]:


#Connect to google drive to get an access to the dataset
# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)


# Importing all the important libraries

# In[2]:


import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Suppress tensorflow warnings. Only log the errors
import logging
tf.get_logger().setLevel(logging.ERROR)


# In[3]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[4]:


#unzip the dataset
# get_ipython().system('unzip /content/gdrive/MyDrive/Colab_notebooks/Melanoma_detection/CNN_assignment.zip -d /content/gdrive/MyDrive/Colab_notebooks/Melanoma_detection/CNN_assignment/')


# In[5]:


# Defining the path for train and test images
data_dir_train = pathlib.Path("/home/sanghyuk.kim001/MELANOMA/Melanoma-Skin-Cancer-Detection/ISICdb/Train")
data_dir_test = pathlib.Path('/home/sanghyuk.kim001/MELANOMA/Melanoma-Skin-Cancer-Detection/ISICdb/Test')


# In[6]:


image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
print("Images available in train dataset:", image_count_train)
image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
print("Images available in test dataset:", image_count_test)


# In[7]:


#considering some predefined inputs from the assignment
batch_size = 32
img_height = 180
img_width = 180
seed_val = 123


# ### Load using keras.preprocessing

# Use 80% of the images for training, and 20% for validation.

# In[8]:


# Loading the training data
# using seed=123 while creating dataset using tf.keras.preprocessing.image_dataset_from_directory
# resizing images to the size img_height*img_width, while writing the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir_train,
                                                               seed=seed_val,
                                                               validation_split=0.2,
                                                               image_size=(img_height,img_width),
                                                               batch_size=batch_size,
                                                               color_mode='rgb',
                                                               subset='training')


# In[9]:


# Loading the validation data
# using seed=123 while creating dataset using tf.keras.preprocessing.image_dataset_from_directory
# resizing images to the size img_height*img_width, while writing the dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir_train,
                                                             seed=seed_val,
                                                             validation_split=0.2,
                                                             image_size=(img_height,img_width),
                                                             batch_size=batch_size,
                                                             color_mode='rgb',
                                                             subset='validation')


# In[10]:


# List out all the classes of skin cancer and store them in a list. 
class_names = train_ds.class_names
print("The different types of cancer classes are: ")
print(class_names)


# ### Visualize the data
# #### Visualize one instance of all the nine classes present in the dataset

# In[11]:


### visualize one instance of all the nine classes present in the dataset
plt.figure(figsize=(15,15))
for i in range(len(class_names)):
  plt.subplot(3,3,i+1)
  image= plt.imread(str(list(data_dir_train.glob(class_names[i]+'/*.jpg'))[0]))
  plt.title(class_names[i])
  plt.imshow(image)


# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.

# `Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch.
# 
# `Dataset.prefetch()` overlaps data preprocessing and model execution while training.

# In[12]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ### Create the first model
# #### Creating a CNN model, which can accurately detect 9 classes present in the dataset. Using ```layers.experimental.preprocessing.Rescaling``` to normalize pixel values between (0,1). The RGB channel values are in the `[0, 255]` range. 

# In[13]:


# CNN Model - Initial
model=models.Sequential()
# scaling the pixel values from 0-255 to 0-1
model.add(layers.experimental.preprocessing.Rescaling(scale=1./255,input_shape=(img_height,img_width,3)))

# Convolution layer with 32 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 64 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 128 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D())

#Dropout layer with 50% Fraction of the input units to drop.
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))

#Dropout layer with 25% Fraction of the input units to drop.
model.add(layers.Dropout(0.25))

model.add(layers.Dense(len(class_names),activation='softmax'))


# In[14]:


# Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()


# In[15]:


# Training the model
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# In[16]:


#Method to visualize the training results

def visualize_cnn(history, epochs):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(12, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='upper left')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper center')
  plt.title('Training and Validation Loss')
  plt.show()


# In[17]:


#visualize the training results
visualize_cnn(history, epochs)


# ### Observations:
# 1. As the number of epochs increase, the training accuracy increases whereas the validation accuracy increases to a max value of 50-55% and then stalls.
# 2. As the number of epochs increase, the training loss decreases whereas the validation loss decreases in the start but later keeps on increasing.
# 3. Overall, the validation accuracy was around **50-55%** for the model.
# 4. The high training accuracy and low validation accuracy tells us that <mark>**the model is Overfitting and needs tuning**</mark>.

# ### Conclusion: 
# Overfitting can happen due to several reasons, such as:
# 
# - The training data size is too small and does not contain enough data samples to accurately represent all possible input data values.
# - The training data contains large amounts of irrelevant information, also called noisy data.
# - The model trains for too long on a single sample set of data.
# - The model complexity is high, so it learns the noise within the training data.
# 
# For our model, looks like the training data is insufficient. So, let's try to perform some data augmentation strategy to come up with a bigger dataset

# Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset.

# In[18]:


#Performing data augmentation on the training dataset
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# In[19]:


# visualizing the augmentation strategy for one instance of a training image.
plt.figure(figsize=(15, 15))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[1].numpy().astype("uint8"))
    plt.axis("off")


# Now, using this augmented data, let's come up with a new model

# In[20]:


# CNN Model with data augmentation
model=models.Sequential()
# scaling the pixel values from 0-255 to 0-1
model.add(layers.experimental.preprocessing.Rescaling(scale=1./255,input_shape=(img_height,img_width,3)))

# adding the augmentation layer before the convolution layer
model.add(data_augmentation)

# Convolution layer with 32 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(32,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 64 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(64,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 128 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(128,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

#Dropout layer with 50% Fraction of the input units to drop.
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))

#Dropout layer with 25% Fraction of the input units to drop.
model.add(layers.Dropout(0.25))

model.add(layers.Dense(len(class_names),activation='softmax'))


# In[21]:


# Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()


# In[22]:


# Training the model
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# In[23]:


#visualize the results of our model after the data augmentation
visualize_cnn(history, epochs)


# #### Observation
# 
# 1. It can be observed that the gap between the training accuracy and the validation accuracy is much less now when compared to the initial model.
# 2. The same can be observed with the training loss and the validation loss.
# 3. The overall accuracy also looks improved but only by a little.
# 4. This implies that the <mark>**overfitting of the model is greatly reduced**</mark> when compared to the initial model but <mark>the overall accuracy isn't really great.</mark>

# #### Class Distribution
# 
# Let's check the class distribution to see if all the classes(cancer types) are equally distributed

# In[24]:


#plot the images to check if all the cancer types are equally distributed
fig = plt.figure(figsize=(12,8))
ax = fig.add_axes([0,0,1,1])
x=[]
y=[]
for i in range(len(class_names)):
  x.append(class_names[i])
  y.append(len(list(data_dir_train.glob(class_names[i]+'/*.jpg'))))

ax.bar(x,y)
ax.set_ylabel('Numbers of images')
ax.set_title('Class distribution of the different cancer types')
plt.xticks(rotation=45)
plt.show()


# In[25]:


print("Number of samples for each class: ")
for i in range(len(class_names)):
  print(class_names[i],' - ',len(list(data_dir_train.glob(class_names[i]+'/*.jpg'))))


# #### Observation
# 
# 1. Class imbalance is observed. It can be seen that some classes have proportionately higher number of samples compared to the others. Class imbalance can have a detrimental effect on the final model quality.
# 2. <mark>The class ***seborrheic keratosis*** has the least number of samples with just 77 images.</mark>
# 3. <mark>The class ***pigmented benign keratosis*** has the highest number of samples with 462 images.</mark>

# To rectify this class imbalance, Augmentor library can be used to artificially generate newer samples.

# In[26]:


#Install Augmentor
# get_ipython().system('pip install Augmentor')


# In[27]:


path_to_training_dataset="/home/sanghyuk.kim001/MELANOMA/Melanoma-Skin-Cancer-Detection/ISICdb/Train/"
import Augmentor
for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + i)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500)  #Adding 500 samples per class to make sure that none of the classes are sparse


# Augmentor has stored the augmented images in the output sub-directory of each of the sub-directories of skin cancer types.. Lets take a look at total count of augmented images.

# In[28]:


data_dir_train = pathlib.Path("/home/sanghyuk.kim001/MELANOMA/Melanoma-Skin-Cancer-Detection/ISICdb/Train")
image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print("Newly generated images with the Augmentor library:", image_count_train)


# Lets see the distribution of augmented data after adding new images to the original training data.

# In[29]:


#plot the images to check if all the cancer types are equally distributed
fig = plt.figure(figsize=(14,8))
ax = fig.add_axes([0,0,1,1])
x=[]
y=[]
for i in range(len(class_names)):
  x.append(class_names[i])
  y.append(len(list(data_dir_train.glob(class_names[i]+'/*.jpg'))) + len(list(data_dir_train.glob(class_names[i]+'/output/*.jpg'))))

ax.bar(x,y)
ax.set_ylabel('Numbers of images')
ax.set_title('Class distribution of the different cancer types(after using Augmentor)')
plt.xticks(rotation=45)
plt.show()


# #### Observation:
# 
# We have added 500 images to all the classes to maintain some class balance. Thus, now there are enough samples to analyze our model.

# Let's create a training and validation dataset with our new samples

# In[30]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=seed_val,
  validation_split = 0.2,
  subset = "training",
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[31]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_train,
  seed=seed_val,
  validation_split = 0.2,
  subset = "validation",
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[32]:


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Train the model on the data created using Augmentor

# In[33]:


# CNN Model on the data created using Augmentor
model=models.Sequential()
# scaling the pixel values from 0-255 to 0-1
model.add(layers.experimental.preprocessing.Rescaling(scale=1./255,input_shape=(img_height,img_width,3)))

# Convolution layer with 32 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(32,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 64 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(64,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

# Convolution layer with 128 features, 3x3 filter and relu activation with 2x2 pooling
model.add(layers.Conv2D(128,(3,3),padding = 'same',activation='relu'))
model.add(layers.MaxPooling2D())

#Dropout layer with 50% Fraction of the input units to drop.
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))

#Dropout layer with 25% Fraction of the input units to drop.
model.add(layers.Dropout(0.25))

model.add(layers.Dense(len(class_names),activation='softmax'))


# In[34]:


# Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()


# In[35]:


# Training the model
epochs = 20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# In[36]:


#visualize the results of model
visualize_cnn(history, epochs)


# ### Observation
# 
# 1. With the increase in the training accuracy over time, where as the validation accuracy also increases.
# 2. The validation loss also decreases over time.
# 3. The gap between training accuracy and validation accuracy has decreased significantly compared to the previous model, and it has achieved around 84% accuracy on the validation set.
# 
# 
# Finally, <mark>**Class rebalancing improved the overall accuracy and also reduced the overall loss**.</mark>

# In[ ]:




