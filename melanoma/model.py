
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
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
import dataframe_image as dfi

import melanoma as mel


class Model:
    # img_height, img_width, class_names
    
    def __init__(self, CFG, train_images, train_labels, val_images, val_labels, test_images, test_labels):
        
        # self.train_ds = train_ds
        # self.val_ds = val_ds
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.CFG = CFG
        # self.img_size = (CFG['img_height'], CFG['img_width'])
        # self.image_shape = (CFG['img_height'], CFG['img_width'], 3)
        # self.num_classes = CFG['num_classes']
        
        
        

        # self.CFG_last_trainable_layers = self.CFG['last_trainable_layers']
        # self.CFG_early_stopper_patience = self.CFG['stopper_patience']
        # self.CFG_epochs = self.CFG['epochs']
        # self.CFG_batch_size = self.CFG['batch_size']

    def build_model(self,
        base_model,
        base_model_name,
        model_optimizer,
        raw_model = False,
        last_trainable_layers = None,
        model_loss = 'sparse_categorical_crossentropy'):
            if last_trainable_layers is None:
                last_trainable_layers = self.CFG['last_trainable_layers']
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
    
    def fit_model(self, model, model_name, trainimages, trainlabels, validationimages, validationlabels):
        data_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=self.CFG['ROTATION_RANGE'],  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = self.CFG['ZOOM_RANGE'], # Randomly zoom image 
            width_shift_range=self.CFG['WSHIFT_RANGE'],  # randomly shift images horizontally (fraction of total width)
            height_shift_range=self.CFG['HSHIFT_RANGE'],  # randomly shift images vertically (fraction of total height)
            horizontal_flip=self.CFG['HFLIP'],  # randomly flip images
            vertical_flip=self.CFG['VFLIP'] # randomly flip images
        )  
        snapshot_path = self.CFG['snapshot_path']
        early_stopper_patience = self.CFG['stopper_patience']
        epochs = self.CFG['epochs']
        batch_size = self.CFG['batch_size']
        # tf.function - decorated function tried to create variables on non-first call'. 
        # tf.config.run_functions_eagerly(self.CFG['run_functions_eagerly']) # otherwise error

        print(f'Fitting {model_name} model...')
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        
        # cb_early_stopper_loss = EarlyStopping(monitor = 'loss', patience = early_stopper_patience)
        print("model_name: " + f"{model_name}")
        cb_checkpointer  = ModelCheckpoint(
            filepath=f'{snapshot_path}/{model_name}.hdf5',
            # filepath=f'{snapshot_path}/{model_name}.hdf5',
            # filepath = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            # filepath = 'snapshot/{model_name}_{epochs:-2d}-{val_loss:.2f}.hdf5',
            # filepath = CFG['path_model']+'ResNet50-{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor  = 'val_loss',
            save_best_only=True, 
            mode='min'
        )

        # callbacks_list = [cb_checkpointer, cb_early_stopper_val_loss, silent_training_callback()]
        extracallbacks = self.CFG['callbacks']

        history = model.fit(
            data_gen.flow(trainimages, trainlabels, batch_size = batch_size, shuffle=True),
            epochs = epochs,
            # validation_data = data_gen.flow(validationimages, validationlabels, batch_size = batch_size),
            validation_data = (validationimages, validationlabels),
            verbose = self.CFG['verbose'],
            steps_per_epoch=trainimages.shape[0] // batch_size,
            callbacks=[cb_checkpointer, extracallbacks], # We can add GCCollectCallback() to save memory
        )

        return history
    

    def evaluate_model_onAll(self, model_name, model_path, network_name, dbpath_KaggleDB, dbpath_HAM10000, dbpath_ISIC2016, dbpath_ISIC2017, dbpath_ISIC2018, \
        dbpath_7pointcriteria, dbpath_ISIC2020, leaderboard_only = False):

        commondata = mel.CommonData()
        
        DBtypes = [db.name for db in mel.DatasetType]
        combined_DBs = [each_model for each_model in DBtypes if(each_model in model_name)]
        assert len(combined_DBs) >= 1

        # ISIC2020
        trainimages, testimages, validationimages, \
			trainlabels, _, validationlabels, num_classes, testimages_id = pickle.load(open(dbpath_ISIC2020, 'rb'))
        assert testimages.shape[0] == 10982
        print('Testing on ISIC2020 DB')

        model = load_model(model_path+'/'+model_name + '.hdf5')
        target_network = model.layers[0].name

        test_pred, test_pred_classes = self.predict_testimages(
            model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2020.name, \
                testimages = testimages
        )

        import csv

        # field names
        fields = ['image_name', 'target']
        
        # name of csv file
        if not os.path.exists(f'{model_path}/leaderboard/{target_network}'):
            os.makedirs(f'{model_path}/leaderboard/{target_network}', exist_ok=True)
        filename = f'{model_path}/leaderboard/{target_network}/{model_name}_ISIC2020leaderboard.csv'
        
        

        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
        
            # writing the fields
            csvwriter.writerow(fields)
        
            assert len(testimages_id) == len(test_pred_classes)
            assert len(testimages_id) == len(test_pred)
            # writing the data rows
            for idx, i in enumerate(testimages_id):

                csvwriter.writerow([i, test_pred[idx][1]])

        


        if leaderboard_only == False:

            # 7 point criteria Testing
            trainimages, testimages, validationimages, \
                trainlabels, testlabels, validationlabels, num_classes = pickle.load(open(dbpath_7pointcriteria, 'rb'))
            assert testimages.shape[0] == 395
            assert len(testlabels) == 395
            print('Testing on 7-point-criteria DB')
            model, _, _ = self.evaluate_model(
                model_name=model_name,
                model_path=model_path,
                target_db=mel.DatasetType._7_point_criteria.name,
                trainimages=trainimages,
                trainlabels=trainlabels,
                validationimages=validationimages,
                validationlabels=validationlabels,
                testimages=testimages,
                testlabels=testlabels,
                )
            target_network = model.layers[0].name

            train_pred, train_pred_classes, test_pred, test_pred_classes = self.computing_prediction(
                model = model, model_name = model_name, target_db=mel.DatasetType._7_point_criteria.name, \
                    trainimages = trainimages, testimages = testimages
            )
            test_report = self.model_report(
                model_name = model_name, model_path=model_path, target_db=mel.DatasetType._7_point_criteria.name, \
                    target_network = target_network, trainlabels = trainlabels, train_pred_classes = train_pred_classes, \
                        testlabels = testlabels, test_pred_classes = test_pred_classes
            )

            _7_point_criteria_perf = {
                'y_pred': test_pred_classes.tolist(),
                'accuracy': test_report['accuracy'],
                'precision': test_report['macro avg']['precision'],
                'sensitivity': test_report['Malignant']['recall'],
                'specificity': test_report['Benign']['recall'],
                'f1-score': test_report['macro avg']['f1-score'],
            }

            # Kaggle MB Testing
            trainimages, testimages, validationimages, \
                trainlabels, testlabels, validationlabels, num_classes = pickle.load(open(dbpath_KaggleDB, 'rb'))
            print('Testing on Kaggle DB')
            model, test_loss_KMB, test_acc_KMB = self.evaluate_model(
                model_name=model_name,
                model_path=model_path,
                target_db=mel.DatasetType.KaggleMB.name,
                trainimages=trainimages,
                trainlabels=trainlabels,
                validationimages=validationimages,
                validationlabels=validationlabels,
                testimages=testimages,
                testlabels=testlabels,
                )
            
            
            train_pred, train_pred_classes, test_pred, test_pred_classes = self.computing_prediction(
                model = model, model_name = model_name, target_db=mel.DatasetType.KaggleMB.name, \
                    trainimages = trainimages, testimages = testimages
            )
            

            test_report = self.model_report(
                model_name = model_name, model_path=model_path, target_db=mel.DatasetType.KaggleMB.name, \
                    target_network = target_network, trainlabels = trainlabels, train_pred_classes = train_pred_classes, \
                        testlabels = testlabels, test_pred_classes = test_pred_classes
            )
            
            KaggleMB_perf = {
                'y_pred': test_pred_classes.tolist(),
                'accuracy': test_report['accuracy'],
                'precision': test_report['macro avg']['precision'],
                'sensitivity': test_report['Malignant']['recall'],
                'specificity': test_report['Benign']['recall'],
                'f1-score': test_report['macro avg']['f1-score'],
            }

            # HAM10000 Testing
            trainimages, testimages, validationimages, \
                trainlabels, testlabels, validationlabels, num_classes = pickle.load(open(dbpath_HAM10000, 'rb'))
            print('Testing on HAM10000')
            model, _, _ = self.evaluate_model(
                model_name=model_name,
                model_path=model_path,
                target_db=mel.DatasetType.HAM10000.name,
                trainimages=trainimages,
                trainlabels=trainlabels,
                validationimages=validationimages,
                validationlabels=validationlabels,
                testimages=testimages,
                testlabels=testlabels,
                )
            train_pred, train_pred_classes, test_pred, test_pred_classes = self.computing_prediction(
                model = model, model_name = model_name, target_db=mel.DatasetType.HAM10000.name, \
                    trainimages = trainimages, testimages = testimages
            )
            test_report = self.model_report(
                model_name = model_name, model_path=model_path, target_db=mel.DatasetType.HAM10000.name, \
                    target_network = target_network, trainlabels = trainlabels, train_pred_classes = train_pred_classes, \
                        testlabels = testlabels, test_pred_classes = test_pred_classes
            )

            HAM10000_perf = {
                'y_pred': test_pred_classes.tolist(),
                'accuracy': test_report['accuracy'],
                'precision': test_report['macro avg']['precision'],
                'sensitivity': test_report['Malignant']['recall'],
                'specificity': test_report['Benign']['recall'],
                'f1-score': test_report['macro avg']['f1-score'],
            }

            # ISIC2016 Testing
            trainimages, testimages, validationimages, \
                trainlabels, testlabels, validationlabels, num_classes = pickle.load(open(dbpath_ISIC2016, 'rb'))
            assert testimages.shape[0] == 379
            assert len(testlabels) == 379
            print('Testing on ISIC2016')
            model, _, _ = self.evaluate_model(
                model_name=model_name,
                model_path=model_path,
                target_db=mel.DatasetType.ISIC2016.name,
                trainimages=trainimages,
                trainlabels=trainlabels,
                validationimages=validationimages,
                validationlabels=validationlabels,
                testimages=testimages,
                testlabels=testlabels,
                )
            train_pred, train_pred_classes, test_pred, test_pred_classes = self.computing_prediction(
                model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2016.name, \
                    trainimages = trainimages, testimages = testimages
            )
            test_report = self.model_report(
                model_name = model_name, model_path=model_path, target_db=mel.DatasetType.ISIC2016.name, \
                    target_network = target_network, trainlabels = trainlabels, train_pred_classes = train_pred_classes, \
                        testlabels = testlabels, test_pred_classes = test_pred_classes
            )

            ISIC2016_perf = {
                'y_pred': test_pred_classes.tolist(),
                'accuracy': test_report['accuracy'],
                'precision': test_report['macro avg']['precision'],
                'sensitivity': test_report['Malignant']['recall'],
                'specificity': test_report['Benign']['recall'],
                'f1-score': test_report['macro avg']['f1-score'],
            }

            # ISIC2017 Testing
            trainimages, testimages, validationimages, \
                trainlabels, testlabels, validationlabels, num_classes = pickle.load(open(dbpath_ISIC2017, 'rb'))
            assert testimages.shape[0] == 600
            assert len(testlabels) == 600
            print('Testing on ISIC2017')
            model, _, _ = self.evaluate_model(
                model_name=model_name,
                model_path=model_path,
                target_db=mel.DatasetType.ISIC2017.name,
                trainimages=trainimages,
                trainlabels=trainlabels,
                validationimages=validationimages,
                validationlabels=validationlabels,
                testimages=testimages,
                testlabels=testlabels,
                )
            train_pred, train_pred_classes, test_pred, test_pred_classes = self.computing_prediction(
                model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2017.name, \
                    trainimages = trainimages, testimages = testimages
            )
            test_report = self.model_report(
                model_name = model_name, model_path=model_path, target_db=mel.DatasetType.ISIC2017.name, \
                    target_network = target_network, trainlabels = trainlabels, train_pred_classes = train_pred_classes, \
                        testlabels = testlabels, test_pred_classes = test_pred_classes
            )

            ISIC2017_perf = {
                'y_pred': test_pred_classes.tolist(),
                'accuracy': test_report['accuracy'],
                'precision': test_report['macro avg']['precision'],
                'sensitivity': test_report['Malignant']['recall'],
                'specificity': test_report['Benign']['recall'],
                'f1-score': test_report['macro avg']['f1-score'],
            }

            # ISIC2018 Testing
            trainimages, testimages, validationimages, \
                trainlabels, testlabels, validationlabels, num_classes = pickle.load(open(dbpath_ISIC2018, 'rb'))
            assert testimages.shape[0] == 1512
            assert len(testlabels) == 1512
            print('Testing on ISIC2018')
            model, _, _ = self.evaluate_model(
                model_name=model_name,
                model_path=model_path,
                target_db=mel.DatasetType.ISIC2018.name,
                trainimages=trainimages,
                trainlabels=trainlabels,
                validationimages=validationimages,
                validationlabels=validationlabels,
                testimages=testimages,
                testlabels=testlabels,
                )
            train_pred, train_pred_classes, test_pred, test_pred_classes = self.computing_prediction(
                model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2018.name, \
                    trainimages = trainimages, testimages = testimages
            )
            test_report = self.model_report(
                model_name = model_name, model_path=model_path, target_db=mel.DatasetType.ISIC2018.name, \
                    target_network = target_network, trainlabels = trainlabels, train_pred_classes = train_pred_classes, \
                        testlabels = testlabels, test_pred_classes = test_pred_classes
            )

            ISIC2018_perf = {
                'y_pred': test_pred_classes.tolist(),
                'accuracy': test_report['accuracy'],
                'precision': test_report['macro avg']['precision'],
                'sensitivity': test_report['Malignant']['recall'],
                'specificity': test_report['Benign']['recall'],
                'f1-score': test_report['macro avg']['f1-score'],
            }

            

            final_perf = {
                'dataset': combined_DBs,
                'classifier': network_name,
                'Parameters': model.count_params(),
                'HAM10000': HAM10000_perf,
                'KaggleMB': KaggleMB_perf,
                'ISIC2016': ISIC2016_perf,
                'ISIC2017': ISIC2017_perf,
                'ISIC2018': ISIC2018_perf,
                '_7_point_criteria': _7_point_criteria_perf,
                'Filesize': int(os.stat(model_path+'/'+model_name + '.hdf5').st_size / (1024 * 1024)),

            }

            # Into snapshot_path
            if not os.path.exists(f'{model_path}/performance/{target_network}'):
                os.makedirs(f'{model_path}/performance/{target_network}', exist_ok=True)
            
            file_json = open(f'{model_path}/performance/{target_network}/{model_name}_metrics.json', "w")
    
            json.dump(final_perf, file_json, indent = 6)
            
            file_json.close()

            return final_perf

        

    
    def evaluate_model(self,
    model_name,
    model_path,
    target_db,
    trainimages,
    trainlabels,
    validationimages,
    validationlabels,
    testimages,
    testlabels
    ):
        print(f'Evaluating {model_name} model on {target_db}...\n')
        # model = load_model(f'./model/{model_name}.hdf5') # Loads the best fit model
        model = load_model(model_path+'/'+model_name + '.hdf5')

        # print("Train loss = {}  ;  Train accuracy = {:.2%}\n".format(*model.evaluate(trainimages, trainlabels, verbose = self.CFG['verbose'])))

        # print("Validation loss = {}  ;  Validation accuracy = {:.2%}\n".format(*model.evaluate(validationimages, validationlabels, verbose = self.CFG['verbose'])))

        test_loss, test_acc = model.evaluate(testimages, testlabels, verbose = self.CFG['verbose'])
        print(f"Test loss = {test_loss}  ;  Test accuracy = {test_acc:.2%}")

        return (model, test_loss, test_acc)
	
    def computing_prediction(self, model, model_name, target_db, trainimages, testimages):
        print(f'Computing predictions for {model_name} on {target_db}...')
        # train_pred = model.predict(trainimages)
        # train_pred_classes = np.argmax(train_pred,axis = 1)
        train_pred = 0
        train_pred_classes = 0
        test_pred = model.predict(testimages)
        # Convert predictions classes to one hot vectors
        test_pred_classes = np.argmax(test_pred,axis = 1)

        return train_pred, train_pred_classes, test_pred, test_pred_classes
    
    def predict_testimages(self, model, model_name, target_db, testimages):
        print(f'Computing predictions for {model_name} on {target_db}...')
       
        test_pred = model.predict(testimages)
        # Convert predictions classes to one hot vectors
        test_pred_classes = np.argmax(test_pred,axis = 1)

        

        return test_pred, test_pred_classes

    def model_report(self,
        model_path,
        model_name,
        target_db,
        target_network,
        trainlabels,
        train_pred_classes,
        testlabels,
        test_pred_classes,
        fontsize = 13
    ):
        label_substitution = {
			0.0: 'Benign',
			1.0: 'Malignant'
		}
        
        # trainlabels_digit = np.argmax(trainlabels, axis=1)
        testlabels_digit = np.argmax(testlabels, axis=1)

        # train_report = classification_report(trainlabels_digit, train_pred_classes, target_names=label_substitution.values(), output_dict = True)
        test_report = classification_report(testlabels_digit, test_pred_classes, target_names=label_substitution.values(), output_dict = True)

        print(f'Model report for {model_name} model ->\n\n')
        # print("Train Report :\n", train_report)
        print("Test Report :\n", test_report)

        cm = confusion_matrix(testlabels_digit, test_pred_classes)
        
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))
        df_cm = pd.DataFrame(cm, index=label_substitution.values(), columns=label_substitution.values())

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, cmap='Blues')
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label', fontsize=fontsize)
        plt.xlabel('Predicted label', fontsize=fontsize)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.title(f'Confusion Matrix of ({model_name}) on {target_db}', fontsize=fontsize)
        # plt.show()
        if not os.path.exists(f'{model_path}/performance/{target_network}'):
            os.makedirs(f'{model_path}/performance/{target_network}', exist_ok=True)
        plt.savefig(f'{model_path}/performance/{target_network}/{model_name}_confusion1.png')
        conf_mat_df = pd.crosstab(testlabels_digit, test_pred_classes, rownames=['GT Label'],colnames=['Predict'])
        df_styled = conf_mat_df.style.background_gradient()
        dfi.export(df_styled, f"{model_path}/performance/{target_network}/{model_name}_confusion2.png", table_conversion="matplotlib")
        # conf_mat_df.dfi.export(f"{model_path}/performance/{model_name}_confusion2.png")

        return test_report

	
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