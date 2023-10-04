import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# import Model
from .model import Model

class CNN(Model):

    def __init__(self, img_height, img_width, class_names):
         self.img_height = img_height
         self.img_width = img_width
         self.class_names = class_names

    def CNN_3conv(self, img_height, img_width, class_names):
            # ### Create the first model
            # #### Creating a CNN model, which can accurately detect 9 classes present in the dataset.
            # Using ```layers.experimental.preprocessing.Rescaling``` to normalize pixel values between (0,1). The RGB channel values are in the `[0, 255]` range. 


            # CNN Model - Initial
            model=models.Sequential()
            # scaling the pixel values from 0-255 to 0-1
            ##Todo: change img_height, img_width to get from loadTrainData return values (tf.data.Dataset)
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

            return model
    
    def trainData(self, train_ds, val_ds, img_height, img_width, class_names, epochs=20):
        super().__init__(train_ds, val_ds, epochs)
        model = self.CNN_3conv(img_height, img_width, class_names)
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