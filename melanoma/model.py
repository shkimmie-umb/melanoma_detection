
# Superclass
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Activation, Dropout, BatchNormalization,
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, add
)
from keras.layers.merge import concatenate

from tensorflow.python.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from collections import Counter

from warnings import filterwarnings

import numpy as np

from .callback import Callback as silent_training_callback


class Model:
    # img_height, img_width, class_names
    
    def __init__(self, train_ds, val_ds, num_classes, epochs):
        
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.epochs = epochs

        self.CFG = dict(
			batch_size            =  20,   # 8; 16; 32; 64; bigger batch size => moemry allocation issue
			epochs                =  30,   # 5; 10; 20;
			last_trainable_layers =   0,
			verbose               =   1,   # 0; 1
			fontsize              =  14,

			# Images sizes
			img_width             = 150,   # 600 Original
			img_height            = 112,   # 450 Original

			# Images augs
			ROTATION_RANGE        =  90.0,
			ZOOM_RANGE            =   0.1,
			HSHIFT_RANGE          =   0.1,
			WSHIFT_RANGE          =   0.1,
			HFLIP                 = False,
			VFLIP                 = False,

			# Postprocessing
			stopper_patience      =  10,   # 0.01; 0.05; 0.1; 0.2;
			run_functions_eagerly = False
		)

        self.CFG_last_trainable_layers = self.CFG['last_trainable_layers']
        self.CFG_early_stopper_patience = self.CFG['stopper_patience']
        self.CFG_epochs = self.CFG['epochs']
        self.CFG_batch_size = self.CFG['batch_size']

    def build_model(self,
        base_model,
        base_model_name,
        model_optimizer,
        raw_model = False,
        last_trainable_layers = None,
        model_loss = 'sparse_categorical_crossentropy'):
            if last_trainable_layers is None:
                last_trainable_layers = self.CFG_last_trainable_layers
            print(f'Building {base_model_name} model...')

            # We reduce significantly number of trainable parameters by freezing certain layers,
            # excluding from training, i.e. their weights will never be updated
            for layer in base_model.layers:
                layer.trainable = False

            if 0 < last_trainable_layers < len(base_model.layers):
                for layer in base_model.layers[-last_trainable_layers:]:
                    layer.trainable = True

            if raw_model == True:
                model = base_model
            else:
                model = Sequential([
                    base_model,

                    Dropout(0.5),
                    Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.02)),

                    Dropout(0.5),
                    Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.02)) # num classes = 9
                ])

            model.compile(
                optimizer = model_optimizer,
            # loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = CFG['label_smooth_fac']),
                loss = model_loss,
                metrics=['accuracy']
            )
            
            return model
    
    def fit_model(self,
    model,
    model_name,
    trainimages,
    trainlabels,
    validationimages,
    validationlabels,
    data_gen,
    early_stopper_patience = None,
    epochs = None,
    batch_size = None
    ):
        if early_stopper_patience is None:
            early_stopper_patience = self.CFG_early_stopper_patience
        if epochs is None:
            epochs = self.CFG_epochs
        if batch_size is None:
            batch_size = self.CFG_batch_size
        # tf.function - decorated function tried to create variables on non-first call'. 
        tf.config.run_functions_eagerly(self.CFG['run_functions_eagerly']) # otherwise error

        print(f'Fitting {model_name} model...')
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = early_stopper_patience)
        cb_checkpointer  = ModelCheckpoint(
            filepath=f'model/{model_name}.hdf5',
        # filepath = CFG['path_model']+'ResNet50-{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor  = 'val_loss',
            save_best_only=True, 
            mode='min'
        )

        callbacks_list = [cb_checkpointer, cb_early_stopper]

        history = model.fit(
            data_gen.flow(trainimages, trainlabels, batch_size = batch_size),
            epochs = epochs,
            validation_data = data_gen.flow(validationimages, validationlabels, batch_size = batch_size),
            verbose = self.CFG['verbose'],
            steps_per_epoch=trainimages.shape[0] // batch_size,
            callbacks=[cb_checkpointer, cb_early_stopper, silent_training_callback()] # We can add GCCollectCallback() to save memory
        )

        return history
    

    def evaluate_model(self,
    model_name,
    trainimages,
    trainlabels,
    validationimages,
    validationlabels,
    testimages,
    testlabels
    ):
        print(f'Evaluating {model_name} model...\n')
        model = load_model(f'./model/{model_name}.hdf5') # Loads the best fit model

        print("Train loss = {}  ;  Train accuracy = {:.2%}\n".format(*model.evaluate(trainimages, trainlabels, verbose = self.CFG['verbose'])))

        print("Validation loss = {}  ;  Validation accuracy = {:.2%}\n".format(*model.evaluate(validationimages, validationlabels, verbose = self.CFG['verbose'])))

        test_loss, test_acc = model.evaluate(testimages, testlabels, verbose = self.CFG['verbose'])
        print(f"Test loss = {test_loss}  ;  Test accuracy = {test_acc:.2%}")

        return (model, test_loss, test_acc)
	
    def computing_prediction(self, model, model_name, trainimages, testimages):
        print(f'Computing predictions for {model_name}...')
        train_pred = model.predict(trainimages)
        train_pred_classes = np.argmax(train_pred,axis = 1)
        test_pred = model.predict(testimages)
        # Convert predictions classes to one hot vectors
        test_pred_classes = np.argmax(test_pred,axis = 1)

        return train_pred, train_pred_classes, test_pred, test_pred_classes



	
    def trainData(self):
		# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB).
		# The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.
		# `Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch.
		# `Dataset.prefetch()` overlaps data preprocessing and model execution while training.
        # 
        pass
        # AUTOTUNE = tf.data.experimental.AUTOTUNE
        # train_ds = train_ds_input.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        # val_ds = val_ds_input.cache().prefetch(buffer_size=AUTOTUNE)
		# cnnmd1 = Model.Model()
		# img_width = 180
		# img_height = 180
		##ToDo: change img size passing logic
        # model = cnnmd1.CNN(img_width, img_height, self.class_names) # Get CNN model to use
		# Compiling the model

        # return history