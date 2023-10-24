import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# import Model
from .model import Model as Base_Model
from .augmentationStrategy import *

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Activation, Dropout, BatchNormalization,
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, add
)
from keras.layers.merge import concatenate

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input


class CNN(Base_Model):

    def __init__(self, img_height, img_width, class_names, num_classes):
        super().__init__(None, None, num_classes, None)
        self.img_height = img_height
        self.img_width = img_width
        self.class_names = class_names

         

    def augmentation(self, img_height, img_width, rotation, zoom):
         return self._augmentation_strategy.augmentation(img_height, img_width, rotation, zoom)
    


    def CNN_3conv(self, img_height, img_width, class_names, augmentation_type):
            # ### Create the first model
            # #### Creating a CNN model, which can accurately detect 9 classes present in the dataset.
            # Using ```layers.experimental.preprocessing.Rescaling``` to normalize pixel values between (0,1). The RGB channel values are in the `[0, 255]` range. 


            # CNN Model - Initial
            model=models.Sequential()
            # scaling the pixel values from 0-255 to 0-1
            ##Todo: change img_height, img_width to get from loadTrainData return values (tf.data.Dataset)
            model.add(layers.experimental.preprocessing.Rescaling(scale=1./255,input_shape=(img_height,img_width,3)))

            if augmentation_type is not None:
                # print(augmentation_type)
                model.add(Augmentation(augmentation_type).augmentation(img_height, img_width))
            

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

            return model
    
    def resnet50(self, ResNet50_name):
        
        resnet50_out = ResNet50(
            include_top=False,
            input_shape=(self.CFG['img_height'], self.CFG['img_width'], 3),
            pooling = 'avg',
            weights='imagenet'
        )

        Resnet50_model = super().build_model(
            base_model = resnet50_out,
            base_model_name = ResNet50_name,
            model_optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        )

        return Resnet50_model
    
    def trainData(self, train_ds, val_ds, img_height, img_width, class_names, augmentation_type, epochs=20):
        super().__init__(train_ds, val_ds, epochs)
        # augmentation_type = simple_augmentation()
        model = self.CNN_3conv(img_height, img_width, class_names, augmentation_type)
        model.compile(optimizer='adam',
		              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		              metrics=['accuracy'])
        model.summary()

        

		# Training the model
		# epochs = 20
        history = model.fit(
		  train_ds,
		  validation_data=val_ds,
		  epochs=epochs
		)

        return history
    
