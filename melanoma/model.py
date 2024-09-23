
# Superclass
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import torch

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
import time
from tempfile import TemporaryDirectory

import melanoma as mel


class Model:
    def __init__(self):
        
        pass

    @staticmethod
    def train_model(conf, network, data, dataset_sizes):
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(network.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(conf['epochs']):
                print(f"Epoch {epoch}/{conf['epochs'] - 1}")
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['Train', 'Val']:
                    if phase == 'Train':
                        network.train()  # Set model to training mode
                    else:
                        network.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in data[phase]:
                        inputs = inputs.to(conf['device'])
                        labels = labels.to(conf['device'])

                        # zero the parameter gradients
                        conf['optimizer'].zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'Train'):
                            outputs = network(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = conf['criterion'](outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'Train':
                                loss.backward()
                                conf['optimizer'].step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'Train':
                        conf['scheduler'].step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == 'Val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(network.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            network.load_state_dict(torch.load(best_model_params_path))
            isExist = os.path.exists(conf['snapshot_path'])
            if not isExist :
                os.makedirs(conf['snapshot_path'])
            else:
                pass
            torch.save(network.state_dict(), os.path.join(conf['snapshot_path'], f"{conf['model_file_name']}.pt"))
        return network
    
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