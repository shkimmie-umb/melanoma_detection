
# Superclass
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, Activation, Dropout, BatchNormalization,
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D, add
)
# from keras.layers.merge import concatenate

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
# from keras.preprocessing.image import img_to_array

import gc

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
import dataframe_image as dfi
import random

import json
import openpyxl
import itertools
import glob
import pathlib

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
    
    @staticmethod
    def fit_model( CFG, model, model_name, trainimages, trainlabels, validationimages, validationlabels):
        def slice_data(images, labels, idxes, batch_size):
            # start = idx
            # end = start + batch_size
            from operator import itemgetter
            assert len(idxes) > 1 and batch_size > 1
            
            batch_idxes = random.choices(idxes, k=batch_size)

            sliced_images = images[batch_idxes]
            sliced_labels = labels[batch_idxes]

            # sliced_images = images[start:end]
            # sliced_labels = labels[start:end]

            assert len(sliced_images) == len(sliced_labels)

            img_array = []
            
            for idx, img in enumerate(sliced_images):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                img_array.append(decoded_img)
            img_array = np.array(img_array) # Convert list to numpy

            return (img_array, sliced_labels)

        def batch_generator(images, labels, batch_size):
            while True:

                # index= random.randint(0, len(images)-1)
                index = np.arange(len(images)-1)
                reordered_indexes = np.random.permutation(index)
                yield slice_data(images, labels, reordered_indexes, batch_size)

        def convert_imgs(images):
            img_array = []
            for idx, img in enumerate(images):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                img_array.append(decoded_img)
            img_array = np.array(img_array) # Convert list to numpy

            return img_array
        
        valimg_array = []
            
        for idx, img in enumerate(validationimages):
            decoded_img = img_to_array(mel.Parser.decode(img))
            decoded_img = mel.Preprocess.normalizeImg(decoded_img)
            valimg_array.append(decoded_img)
        valimg_array = np.array(valimg_array)

        data_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=CFG['ROTATION_RANGE'],  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = CFG['ZOOM_RANGE'], # Randomly zoom image 
            width_shift_range=CFG['WSHIFT_RANGE'],  # randomly shift images horizontally (fraction of total width)
            height_shift_range=CFG['HSHIFT_RANGE'],  # randomly shift images vertically (fraction of total height)
            horizontal_flip=CFG['HFLIP'],  # randomly flip images
            vertical_flip=CFG['VFLIP'], # randomly flip images
            # rescale=1./255
        )  
        snapshot_path = CFG['snapshot_path']
        early_stopper_patience = CFG['stopper_patience']
        
        
        # tf.function - decorated function tried to create variables on non-first call'. 
        # tf.config.run_functions_eagerly(self.CFG['run_functions_eagerly']) # otherwise error


        print(f'Fitting {model_name} model...')
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        
        # cb_early_stopper_loss = EarlyStopping(monitor = 'loss', patience = early_stopper_patience)
        print("model_name: " + f"{model_name}")
        cb_checkpointer  = ModelCheckpoint(
            filepath=f'{snapshot_path}/{model_name}.hdf5',
            monitor  = 'val_loss',
            save_best_only=True, 
            mode='min'
        )

        # callbacks_list = [cb_checkpointer, cb_early_stopper_val_loss, silent_training_callback()]
        extracallbacks = CFG['callbacks']

        # steps_per_epoch = len(trainimages) // CFG['batch_size']
        my_batch_generator = batch_generator(trainimages, trainlabels, CFG['batch_size'])

        # trainimages = convert_imgs(trainimages)

        history = model.fit(
            # data_gen.flow(trainimages, trainlabels, batch_size = CFG['batch_size'], shuffle=True),
            my_batch_generator,
            epochs = CFG['epochs'],
            # validation_data = data_gen.flow(validationimages, validationlabels, batch_size = batch_size),
            validation_data = (valimg_array, validationlabels),
            verbose = CFG['verbose'],
            steps_per_epoch=len(trainimages) // CFG['batch_size'],
            callbacks=[cb_checkpointer, extracallbacks], # We can add GCCollectCallback() to save memory
        )

        
        gc.collect()

        return history
    
    @staticmethod
    def evaluate_leaderboard(model_name, model_path, dbpath, dbname_ISIC2020):
        
        DBtypes = [db.name for db in mel.DatasetType]
        combined_DBs = [each_model for each_model in DBtypes if(each_model in model_name)]
        assert len(combined_DBs) >= 1

        # ISIC2020
        # trainimages, testimages, validationimages, \
		# 	trainlabels, _, validationlabels, num_classes, testimages_id = pickle.load(open(dbpath_ISIC2020, 'rb'))
        traindata, validationdata, testdata = mel.parser_ISIC2020.open_H5(os.path.join(dbpath, dbname_ISIC2020))
        assert len(testdata['testimages']) == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['testimages']
        assert len(testdata['testids']) == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['testimages']
        print('Testing on ISIC2020 DB')

        model = load_model(model_path+'/'+model_name)
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.predict_testimages(
            model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2020.name, \
                testimages = testdata['testimages']
        )

        import csv

        # field names
        fields = ['image_name', 'target']
        
        # name of csv file
        if not os.path.exists(f'{dbpath}/leaderboard/{target_network}'):
            os.makedirs(f'{dbpath}/leaderboard/{target_network}', exist_ok=True)
        filename = f'{dbpath}/leaderboard/{target_network}/{model_name}_ISIC2020leaderboard.csv'
        
        

        with open(filename, 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
        
            # writing the fields
            csvwriter.writerow(fields)
        
            
            assert len(testdata['testimages']) == len(test_pred_classes)
            assert len(testdata['testimages']) == len(test_pred)
            # assert len(testdata['testlabels']) == len(test_pred_classes)
            # assert len(testdata['testlabels']) == len(test_pred)
            assert len(testdata['testids']) == len(test_pred_classes)
            assert len(testdata['testids']) == len(test_pred)

            # writing the data rows
            for idx, id in enumerate(testdata['testids']):

                csvwriter.writerow([id.item().decode('utf-8'), test_pred[idx][1]])

    def evaluate_model_onAll(model_name, model_path, \
                            dbpath_KaggleDB, dbpath_HAM10000, dbpath_ISIC2016, dbpath_ISIC2017, \
                            dbpath_ISIC2018, dbpath_7pointcriteria):
        model = load_model(model_path+'/'+model_name + '.hdf5')

        target_network = model.layers[0].name

        if os.path.exists(f'{model_path}/performance/{target_network}/{model_name}_metrics.json') is False:

            DBtypes = [db.name for db in mel.DatasetType]
            combined_DBs = [each_model for each_model in DBtypes if(each_model in model_name)]
            assert len(combined_DBs) >= 1

            # ---------- HAM10000 -------------
            HAM10000_perf, model = mel.parser_HAM10000.evaluate(dbpath_HAM10000, model_path, model_name)
            # ---------- KaggleMB -------------
            KaggleMB_perf = mel.parser_KaggleMB.evaluate(dbpath_KaggleDB, model_path, model_name)
            # ---------- ISIC2016 -------------
            ISIC2016_perf = mel.parser_ISIC2016.evaluate(dbpath_ISIC2016, model_path, model_name)
            # ---------- ISIC2017 -------------
            ISIC2017_perf = mel.parser_ISIC2017.evaluate(dbpath_ISIC2017, model_path, model_name)
            # ---------- ISIC2018 -------------
            ISIC2018_perf = mel.parser_ISIC2018.evaluate(dbpath_ISIC2018, model_path, model_name)
            # ---------- 7 point criteria -------------
            _7_point_criteria_perf = mel.parser_7pointdb.evaluate(dbpath_7pointcriteria, model_path, model_name)

            gc.collect()
            

            final_perf = {
                'dataset': combined_DBs,
                'classifier': model.layers[0].name,
                'Parameters': model.count_params(),
                'HAM10000': HAM10000_perf,
                'KaggleMB': KaggleMB_perf,
                'ISIC2016': ISIC2016_perf,
                'ISIC2017': ISIC2017_perf,
                'ISIC2018': ISIC2018_perf,
                '_7_point_criteria': _7_point_criteria_perf,
                'Filesize': int(os.stat(f'{os.path.join(model_path, model_name)}.hdf5').st_size / (1024 * 1024)),

            }

            

            # Into snapshot_path
            if not os.path.exists(f'{model_path}/performance/{target_network}'):
                os.makedirs(f'{model_path}/performance/{target_network}', exist_ok=True)
            
            file_json = open(f'{model_path}/performance/{target_network}/{model_name}_metrics.json', "w")

            json.dump(final_perf, file_json, indent = 6)
            
            file_json.close()
            return final_perf
        else:
            print(f'Skipping {model_name}')

    @staticmethod
    def extract_performances(snapshot_path):
        jsonfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/performance/*/*.json', recursive=True)]))
        jsonnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, jsonfiles))

        final_perf = []

        for idx, j in enumerate(jsonfiles):
            fi = open(j)
            jfile = json.load(fi)
            final_perf.append(jfile)


        # # KaggleMB

        # KaggleMB_maxperf = {
        #     "max_accuracy": max(final_perf, key=(lambda item: item['KaggleMB']['accuracy'])),
        #     "max_precision": max(final_perf, key=(lambda item: item['KaggleMB']['precision'])),
        #     "max_sensitivity": max(final_perf, key=(lambda item: item['KaggleMB']['sensitivity'])),
        #     "max_specificity": max(final_perf, key=(lambda item: item['KaggleMB']['specificity'])),
        #     "max_f1-score": max(final_perf, key=(lambda item: item['KaggleMB']['f1-score'])),
        # }

        # KaggleMB_maxacc = {
        #     "datasets": KaggleMB_maxperf['max_accuracy']['dataset'],
        #     "classifier": KaggleMB_maxperf['max_accuracy']['classifier'],
        #     "acc": KaggleMB_maxperf['max_accuracy']['KaggleMB']['accuracy'],
        # }

        # KaggleMB_maxprec = {
        #     "datasets": KaggleMB_maxperf['max_precision']['dataset'],
        #     "classifier": KaggleMB_maxperf['max_precision']['classifier'],
        #     "acc": KaggleMB_maxperf['max_precision']['KaggleMB']['precision'],
        # }

        # KaggleMB_maxsens = {
        #     "datasets": KaggleMB_maxperf['max_sensitivity']['dataset'],
        #     "classifier": KaggleMB_maxperf['max_sensitivity']['classifier'],
        #     "acc": KaggleMB_maxperf['max_sensitivity']['KaggleMB']['sensitivity'],
        # }

        # KaggleMB_maxspec = {
        #     "datasets": KaggleMB_maxperf['max_specificity']['dataset'],
        #     "classifier": KaggleMB_maxperf['max_specificity']['classifier'],
        #     "acc": KaggleMB_maxperf['max_specificity']['KaggleMB']['specificity'],
        # }

        # KaggleMB_maxf1 = {
        #     "datasets": KaggleMB_maxperf['max_accuracy']['dataset'],
        #     "classifier": KaggleMB_maxperf['max_accuracy']['classifier'],
        #     "acc": KaggleMB_maxperf['max_f1-score']['KaggleMB']['f1-score'],
        # }


        # # HAM10000

        # HAM10000_maxperf = {
        #     "max_accuracy": max(final_perf, key=(lambda item: item['HAM10000']['accuracy'])),
        #     "max_precision": max(final_perf, key=(lambda item: item['HAM10000']['precision'])),
        #     "max_sensitivity": max(final_perf, key=(lambda item: item['HAM10000']['sensitivity'])),
        #     "max_specificity": max(final_perf, key=(lambda item: item['HAM10000']['specificity'])),
        #     "max_f1-score": max(final_perf, key=(lambda item: item['HAM10000']['f1-score'])),
        # }

        # HAM10000_maxacc = {
        #     "datasets": HAM10000_maxperf['max_accuracy']['dataset'],
        #     "classifier": HAM10000_maxperf['max_accuracy']['classifier'],
        #     "acc": HAM10000_maxperf['max_accuracy']['HAM10000']['accuracy'],
        # }

        # HAM10000_maxprec = {
        #     "datasets": HAM10000_maxperf['max_precision']['dataset'],
        #     "classifier": HAM10000_maxperf['max_precision']['classifier'],
        #     "acc": HAM10000_maxperf['max_precision']['HAM10000']['precision'],
        # }

        # HAM10000_maxsens = {
        #     "datasets": HAM10000_maxperf['max_sensitivity']['dataset'],
        #     "classifier": HAM10000_maxperf['max_sensitivity']['classifier'],
        #     "acc": HAM10000_maxperf['max_sensitivity']['HAM10000']['sensitivity'],
        # }

        # HAM10000_maxspec = {
        #     "datasets": HAM10000_maxperf['max_specificity']['dataset'],
        #     "classifier": HAM10000_maxperf['max_specificity']['classifier'],
        #     "acc": HAM10000_maxperf['max_specificity']['HAM10000']['specificity'],
        # }

        # HAM10000_maxf1 = {
        #     "datasets": HAM10000_maxperf['max_accuracy']['dataset'],
        #     "classifier": HAM10000_maxperf['max_accuracy']['classifier'],
        #     "acc": HAM10000_maxperf['max_f1-score']['HAM10000']['f1-score'],
        # }

        # # ISIC2016

        # ISIC2016_maxperf = {
        #     "max_accuracy": max(final_perf, key=(lambda item: item['ISIC2016']['accuracy'])),
        #     "max_precision": max(final_perf, key=(lambda item: item['ISIC2016']['precision'])),
        #     "max_sensitivity": max(final_perf, key=(lambda item: item['ISIC2016']['sensitivity'])),
        #     "max_specificity": max(final_perf, key=(lambda item: item['ISIC2016']['specificity'])),
        #     "max_f1-score": max(final_perf, key=(lambda item: item['ISIC2016']['f1-score'])),
        # }

        # ISIC2016_maxacc = {
        #     "datasets": ISIC2016_maxperf['max_accuracy']['dataset'],
        #     "classifier": ISIC2016_maxperf['max_accuracy']['classifier'],
        #     "acc": ISIC2016_maxperf['max_accuracy']['ISIC2016']['accuracy'],
        # }

        # ISIC2016_maxprec = {
        #     "datasets": ISIC2016_maxperf['max_precision']['dataset'],
        #     "classifier": ISIC2016_maxperf['max_precision']['classifier'],
        #     "acc": ISIC2016_maxperf['max_precision']['ISIC2016']['precision'],
        # }

        # ISIC2016_maxsens = {
        #     "datasets": ISIC2016_maxperf['max_sensitivity']['dataset'],
        #     "classifier": ISIC2016_maxperf['max_sensitivity']['classifier'],
        #     "acc": ISIC2016_maxperf['max_sensitivity']['ISIC2016']['sensitivity'],
        # }

        # ISIC2016_maxspec = {
        #     "datasets": ISIC2016_maxperf['max_specificity']['dataset'],
        #     "classifier": ISIC2016_maxperf['max_specificity']['classifier'],
        #     "acc": ISIC2016_maxperf['max_specificity']['ISIC2016']['specificity'],
        # }

        # ISIC2016_maxf1 = {
        #     "datasets": ISIC2016_maxperf['max_accuracy']['dataset'],
        #     "classifier": ISIC2016_maxperf['max_accuracy']['classifier'],
        #     "acc": ISIC2016_maxperf['max_f1-score']['ISIC2016']['f1-score'],
        # }

        # # ISIC2017

        # ISIC2017_maxperf = {
        #     "max_accuracy": max(final_perf, key=(lambda item: item['ISIC2017']['accuracy'])),
        #     "max_precision": max(final_perf, key=(lambda item: item['ISIC2017']['precision'])),
        #     "max_sensitivity": max(final_perf, key=(lambda item: item['ISIC2017']['sensitivity'])),
        #     "max_specificity": max(final_perf, key=(lambda item: item['ISIC2017']['specificity'])),
        #     "max_f1-score": max(final_perf, key=(lambda item: item['ISIC2017']['f1-score'])),
        # }

        # ISIC2017_maxacc = {
        #     "datasets": ISIC2017_maxperf['max_accuracy']['dataset'],
        #     "classifier": ISIC2017_maxperf['max_accuracy']['classifier'],
        #     "acc": ISIC2017_maxperf['max_accuracy']['ISIC2017']['accuracy'],
        # }

        # ISIC2017_maxprec = {
        #     "datasets": ISIC2017_maxperf['max_precision']['dataset'],
        #     "classifier": ISIC2017_maxperf['max_precision']['classifier'],
        #     "acc": ISIC2017_maxperf['max_precision']['ISIC2017']['precision'],
        # }

        # ISIC2017_maxsens = {
        #     "datasets": ISIC2017_maxperf['max_sensitivity']['dataset'],
        #     "classifier": ISIC2017_maxperf['max_sensitivity']['classifier'],
        #     "acc": ISIC2017_maxperf['max_sensitivity']['ISIC2017']['sensitivity'],
        # }

        # ISIC2017_maxspec = {
        #     "datasets": ISIC2017_maxperf['max_specificity']['dataset'],
        #     "classifier": ISIC2017_maxperf['max_specificity']['classifier'],
        #     "acc": ISIC2017_maxperf['max_specificity']['ISIC2017']['specificity'],
        # }

        # ISIC2017_maxf1 = {
        #     "datasets": ISIC2017_maxperf['max_accuracy']['dataset'],
        #     "classifier": ISIC2017_maxperf['max_accuracy']['classifier'],
        #     "acc": ISIC2017_maxperf['max_f1-score']['ISIC2017']['f1-score'],
        # }

        # # ISIC2018

        # ISIC2018_maxperf = {
        #     "max_accuracy": max(final_perf, key=(lambda item: item['ISIC2018']['accuracy'])),
        #     "max_precision": max(final_perf, key=(lambda item: item['ISIC2018']['precision'])),
        #     "max_sensitivity": max(final_perf, key=(lambda item: item['ISIC2018']['sensitivity'])),
        #     "max_specificity": max(final_perf, key=(lambda item: item['ISIC2018']['specificity'])),
        #     "max_f1-score": max(final_perf, key=(lambda item: item['ISIC2018']['f1-score'])),
        # }

        # ISIC2018_maxacc = {
        #     "datasets": ISIC2018_maxperf['max_accuracy']['dataset'],
        #     "classifier": ISIC2018_maxperf['max_accuracy']['classifier'],
        #     "acc": ISIC2018_maxperf['max_accuracy']['ISIC2018']['accuracy'],
        # }

        # ISIC2018_maxprec = {
        #     "datasets": ISIC2018_maxperf['max_precision']['dataset'],
        #     "classifier": ISIC2018_maxperf['max_precision']['classifier'],
        #     "acc": ISIC2018_maxperf['max_precision']['ISIC2018']['precision'],
        # }

        # ISIC2018_maxsens = {
        #     "datasets": ISIC2018_maxperf['max_sensitivity']['dataset'],
        #     "classifier": ISIC2018_maxperf['max_sensitivity']['classifier'],
        #     "acc": ISIC2018_maxperf['max_sensitivity']['ISIC2018']['sensitivity'],
        # }

        # ISIC2018_maxspec = {
        #     "datasets": ISIC2018_maxperf['max_specificity']['dataset'],
        #     "classifier": ISIC2018_maxperf['max_specificity']['classifier'],
        #     "acc": ISIC2018_maxperf['max_specificity']['ISIC2018']['specificity'],
        # }

        # ISIC2018_maxf1 = {
        #     "datasets": ISIC2018_maxperf['max_accuracy']['dataset'],
        #     "classifier": ISIC2018_maxperf['max_accuracy']['classifier'],
        #     "acc": ISIC2018_maxperf['max_f1-score']['ISIC2018']['f1-score'],
        # }


        performance = openpyxl.Workbook()
        performance_ws = performance.active
        performance_ws.title = 'HAM10000'
        performance.create_sheet('ISIC2016')
        performance.create_sheet('ISIC2017')
        performance.create_sheet('ISIC2018')
        performance.create_sheet('KaggleMB')
        performance.create_sheet('7pointcriteria')


        cols = ['Network', 'DB Comb', 'Precision', 'Specificity', 'Sensitivity', 'Accuracy', 'Filesize', 'Parameters']

        HAM10000_ws = performance['HAM10000']
        HAM10000_ws.append(cols)

        for p in final_perf:
            HAM10000_ws.append([p['classifier'], str(p['dataset']), p['HAM10000']['precision'], p['HAM10000']['specificity'], p['HAM10000']['sensitivity'], p['HAM10000']['accuracy'], p['Filesize'], p['Parameters']])
            
        ISIC2016_ws = performance['ISIC2016']
        ISIC2016_ws.append(cols)

        for p in final_perf:
            ISIC2016_ws.append([p['classifier'], str(p['dataset']), p['ISIC2016']['precision'], p['ISIC2016']['specificity'], p['ISIC2016']['sensitivity'], p['ISIC2016']['accuracy'], p['Filesize'], p['Parameters']])

        ISIC2017_ws = performance['ISIC2017']
        ISIC2017_ws.append(cols)

        for p in final_perf:
            ISIC2017_ws.append([p['classifier'], str(p['dataset']), p['ISIC2017']['precision'], p['ISIC2017']['specificity'], p['ISIC2017']['sensitivity'], p['ISIC2017']['accuracy'], p['Filesize'], p['Parameters']])

        ISIC2018_ws = performance['ISIC2018']
        ISIC2018_ws.append(cols)

        for p in final_perf:
            ISIC2018_ws.append([p['classifier'], str(p['dataset']), p['ISIC2018']['precision'], p['ISIC2018']['specificity'], p['ISIC2018']['sensitivity'], p['ISIC2018']['accuracy'], p['Filesize'], p['Parameters']])


        KaggleMB_ws = performance['KaggleMB']
        KaggleMB_ws.append(cols)

        for p in final_perf:
            KaggleMB_ws.append([p['classifier'], str(p['dataset']), p['KaggleMB']['precision'], p['KaggleMB']['specificity'], p['KaggleMB']['sensitivity'], p['KaggleMB']['accuracy'], p['Filesize'], p['Parameters']])

        _7pointcriteria_ws = performance['7pointcriteria']
        _7pointcriteria_ws.append(cols)

        for p in final_perf:
            _7pointcriteria_ws.append([p['classifier'], str(p['dataset']), p['_7_point_criteria']['precision'], p['_7_point_criteria']['specificity'], p['_7_point_criteria']['sensitivity'], p['_7_point_criteria']['accuracy'], p['Filesize'], p['Parameters']])
            
        performance.save(f'{snapshot_path}/performance/performance.xlsx')

        print(f'{snapshot_path}/performance/performance.xlsx' + ' generated')
        

        

    @staticmethod
    def evaluate_model(
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

        test_loss, test_acc = model.evaluate(testimages, testlabels, verbose = 1)
        print(f"Test loss = {test_loss}  ;  Test accuracy = {test_acc:.2%}")

        return (model, test_loss, test_acc)
	
    @staticmethod
    def computing_prediction(model, model_name, target_db, testimages):
        print(f'Computing predictions for {model_name} on {target_db}...')
        # train_pred = model.predict(trainimages)
        # train_pred_classes = np.argmax(train_pred,axis = 1)
        test_pred = model.predict(testimages)
        # Convert predictions classes to one hot vectors
        test_pred_classes = np.argmax(test_pred,axis = 1)

        return test_pred, test_pred_classes
    
    @staticmethod
    def predict_testimages(model, model_name, target_db, testimages):
        print(f'Computing predictions for {model_name} on {target_db}...')

        testimages_decoded = []
        for idx, img in enumerate(testimages):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        test_pred = model.predict(testimages_decoded)
        # Convert predictions classes to one hot vectors
        test_pred_classes = np.argmax(test_pred,axis = 1)

        

        return test_pred, test_pred_classes

    @staticmethod
    def model_report(
        model_path,
        model_name,
        target_db,
        target_network,
        testlabels,
        test_pred_classes,
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

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=13)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=13)
        plt.ylabel('True label', fontsize=13)
        plt.xlabel('Predicted label', fontsize=13)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.title(f'Confusion Matrix of ({model_name}) on {target_db}', fontsize=13)
        # plt.show()
        if not os.path.exists(f'{model_path}/performance/{target_network}'):
            os.makedirs(f'{model_path}/performance/{target_network}', exist_ok=True)
        plt.savefig(f'{model_path}/performance/{target_network}/{model_name}_confusion1.png')
        conf_mat_df = pd.crosstab(testlabels_digit, test_pred_classes, rownames=['GT Label'],colnames=['Predict'])
        df_styled = conf_mat_df.style.background_gradient()
        dfi.export(df_styled, f"{model_path}/performance/{target_network}/{model_name}_confusion2.png", table_conversion="matplotlib")
        # conf_mat_df.dfi.export(f"{model_path}/performance/{model_name}_confusion2.png")

        plt.close()

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