
# Superclass
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import reject
from reject.reject import ClassificationRejector
from reject.reject import confusion_matrix as reject_cm
from reject.utils import generate_synthetic_output

import torch
from tqdm import tqdm

from torchvision.transforms import v2
import torchvision

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
from copy import deepcopy
import torch.nn as nn

import melanoma as mel


class Model:
    def __init__(self):
        
        pass

    @staticmethod
    def train_model(conf, network, data, dataset_sizes):
        since = time.time()
        # dataset_sizes = {}
        # dataset_sizes['Train'] = data['Train'].__len__()
        # dataset_sizes['Val'] = data['Val'].__len__()

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
                    # for inputs, labels in tqdm(data[phase]):
                    for inputs, labels in tqdm(data[phase], total=len(data[phase])):
                        inputs = inputs.to(conf['device'])
                        # labels = torch.flatten(labels).to(conf['device'])
                        labels = labels.to(conf['device'])

                        # zero the parameter gradients
                        conf['optimizer'].zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'Train'):
                            outputs = network(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = conf['criterion'](outputs, labels.type(torch.int64))

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
    def evaluate_model(model, dataloader, device):
        all_labels = []
        all_preds = []
        all_scores = []
        all_paths = []
        CM=0
        model.to(device)
        # model.eval()
        # num_iters = len(dataloaders['Test'].dataset) / dataloaders['Test'].batch_size
        with torch.no_grad():
            for data in tqdm(dataloader['Test']):
                

                images, labels, paths = data
                images = images.to(device)
                # labels = labels.to(device)
                
                outputs = model(images) #file_name
                smx = nn.Softmax(dim=1)
                smx_output = smx(outputs.data)
                # preds = torch.argmax(outputs.data, 1)
                preds = torch.argmax(smx_output, 1)
                CM+=confusion_matrix(labels, preds.cpu(),labels=[0,1])

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
                all_scores.extend(smx_output.cpu().tolist())
                all_paths.extend(paths)

                assert len(all_preds) == len(all_labels) and len(all_labels) == len(all_scores) and \
                    len(all_scores) == len(all_paths)
                
            all_ids = [str(pathlib.Path(path).stem) for path in all_paths]
            tn=CM[0][0]
            tp=CM[1][1]
            fp=CM[0][1]
            fn=CM[1][0]
            acc=np.sum(np.diag(CM)/np.sum(CM))
            sensitivity=tp/(tp+fn)
            precision=tp/(tp+fp)
            
            print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
            print()
            print('Confusion Matirx : ')
            print(CM)
            print('- Sensitivity : ',(tp/(tp+fn))*100)
            print('- Specificity : ',(tn/(tn+fp))*100)
            print('- Precision: ',(tp/(tp+fp))*100)
            print('- NPV: ',(tn/(tn+fn))*100)
            print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
            print()

            


        test_report = classification_report(all_labels, all_preds, target_names=mel.Parser.common_binary_label.values(), output_dict = True)

        performance = {
            'y_labels': all_labels,
            'y_pred': all_preds,
            'y_scores': all_scores,
            'y_ids': all_ids,
            'accuracy': test_report['accuracy'],
            'precision': test_report['macro avg']['precision'],
            'sensitivity': test_report['malignant']['recall'],
            'specificity': test_report['benign']['recall'],
            'f1-score': test_report['macro avg']['f1-score'],
        }
            
                    
        return performance
    
    @staticmethod
    def evaluate_leaderboard(model, model_path, db_path, device):
        all_preds = []
        all_scores = []
        all_ids = []

        data_transform = {
                'Test': v2.Compose([
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        paths = {
            'ISIC2020': os.path.join(db_path, '..', mel.DatasetType.ISIC2020.name, 'ISIC_2020_Test_Input'),
        }

        datafolders = {
            'ISIC2020': {
                'Test': mel.ImageDataset(root_dir=paths['ISIC2020'], transform=data_transform['Test'])
            },
        }

        dataloader = {
            'ISIC2020': {
                'Test': torch.utils.data.DataLoader(datafolders['ISIC2020']['Test'], batch_size=32,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=4, prefetch_factor=2)
            },
            
        }

        model.to(device)
        
        with torch.no_grad():
            for data in tqdm(dataloader['ISIC2020']['Test']):
                

                images, filenames = data
                images = images.to(device)
                
                
                outputs = model(images) #file_name
                smx = nn.Softmax(dim=1)
                smx_output = smx(outputs.data)
                # preds = torch.argmax(outputs.data, 1)
                preds = torch.argmax(smx_output, 1)
                
                all_ids.extend(filenames)
                all_preds.extend(preds.cpu().tolist())
                all_scores.extend(smx_output.cpu().tolist())

                assert len(all_preds) == len(all_ids) and len(all_preds) == len(all_scores) 


        import csv
        # field names
        fields = ['image_name', 'target']
        
        # name of csv file
        csv_path = os.path.join(pathlib.Path(model_path).parent, 'leaderboard')
        full_filename = pathlib.Path(model_path).stem

        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        csv_filename = f'{full_filename}_leaderboard.csv'
        
        
        if os.path.exists(os.path.join(csv_path, csv_filename)) is False:
            with open(os.path.join(csv_path, csv_filename), 'w') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
            
                # writing the fields
                csvwriter.writerow(fields)

                # writing the data rows
                assert len(all_ids) == len(all_scores)
                for idx, label in enumerate(all_scores):
                    csvwriter.writerow([all_ids[idx], all_scores[idx][1]])
        else:
            print(f'Skipping {csv_filename}')
    @staticmethod
    def evaluate_model_onAll(model, model_path, db_path, device):
        data_transform = {
                'Test': v2.Compose([
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        paths = {
            'HAM10000': os.path.join(db_path, mel.DatasetType.HAM10000.name, 'final', 'Test'),
            'ISIC2016': os.path.join(db_path, mel.DatasetType.ISIC2016.name, 'final', 'Test'),
            'ISIC2017': os.path.join(db_path, mel.DatasetType.ISIC2017.name, 'final', 'Test'),
            'ISIC2018': os.path.join(db_path, mel.DatasetType.ISIC2018.name, 'final', 'Test'),
            'KaggleMB': os.path.join(db_path, mel.DatasetType.KaggleMB.name, 'final', 'Test'),
            '_7_point_criteria': os.path.join(db_path, mel.DatasetType._7_point_criteria.name, 'final', 'Test')
        }

        datafolders = {
            'HAM10000': {
                'Test': mel.ImageFolder_filename(paths['HAM10000'], data_transform['Test'])
            },
            'ISIC2016': {
                'Test': mel.ImageFolder_filename(paths['ISIC2016'], data_transform['Test'])
            },
            'ISIC2017': {
                'Test': mel.ImageFolder_filename(paths['ISIC2017'], data_transform['Test'])
            },
            'ISIC2018': {
                'Test': mel.ImageFolder_filename(paths['ISIC2018'], data_transform['Test'])
            },
            '_7_point_criteria': {
                'Test': mel.ImageFolder_filename(paths['_7_point_criteria'], data_transform['Test'])
            },
            'KaggleMB': {
                'Test': mel.ImageFolder_filename(paths['KaggleMB'], data_transform['Test'])
            }
        }

        dataloaders = {
            'HAM10000': {
                'Test': torch.utils.data.DataLoader(datafolders['HAM10000']['Test'], batch_size=32,
                                                            shuffle=False, pin_memory=True
                                                            ,num_workers=4, prefetch_factor=2)
            },
            'ISIC2016': {
                'Test': torch.utils.data.DataLoader(datafolders['ISIC2016']['Test'], batch_size=32,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=4, prefetch_factor=2)
            },
            'ISIC2017': {
                'Test': torch.utils.data.DataLoader(datafolders['ISIC2017']['Test'], batch_size=32,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=4, prefetch_factor=2)
            },
            'ISIC2018': {
                'Test': torch.utils.data.DataLoader(datafolders['ISIC2018']['Test'], batch_size=32,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=4, prefetch_factor=2)
            },
            '_7_point_criteria': {
                'Test': torch.utils.data.DataLoader(datafolders['_7_point_criteria']['Test'], batch_size=32,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=4, prefetch_factor=2)
            },
            'KaggleMB': {
                'Test': torch.utils.data.DataLoader(datafolders['KaggleMB']['Test'], batch_size=32,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=4, prefetch_factor=2)
            },
            
        }

        performances = {
            'HAM10000': None,
            'ISIC2016': None,
            'ISIC2017': None,
            'ISIC2018': None,
            '_7_point_criteria': None,
            'KaggleMB': None,
        }
        json_path = os.path.join(pathlib.Path(model_path).parent, 'performance')
        full_filename = pathlib.Path(model_path).stem
        json_filename = f'{full_filename}_metrics.json'

        if os.path.exists(os.path.join(json_path, json_filename)) is False:
            
            if not os.path.exists(json_path):
                os.makedirs(json_path, exist_ok=True)
            

            classifier_name = pathlib.Path(model_path).parent.name
            

            for db in performances:
                print(f'Evaluating {full_filename} model on {mel.DatasetType[db].name}...\n')
                performances[db] = mel.Model.evaluate_model(model, dataloaders[db], device)    

            DBtypes = [db.name for db in mel.DatasetType]

            def calculate_filesize(model):
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                size_all_mb = (param_size + buffer_size) / 1024**2
                return size_all_mb
            

            used_DBs = [each_model for each_model in DBtypes if(each_model in full_filename)]
            final_perf = {
                'dataset': used_DBs,
                'classifier': classifier_name,
                'Parameters': sum(p.numel() for p in model.parameters()),
                'HAM10000': performances['HAM10000'],
                'KaggleMB': performances['KaggleMB'],
                'ISIC2016': performances['ISIC2016'],
                'ISIC2017': performances['ISIC2017'],
                'ISIC2018': performances['ISIC2018'],
                '_7_point_criteria': performances['_7_point_criteria'],
                'Filesize': '{:.2f}'.format(calculate_filesize(model)),

            }
            # Dump Json
            file_json = open(os.path.join(json_path, json_filename), "w")

            json.dump(final_perf, file_json, indent = 6)
            
            file_json.close()

            
        else:
            print(f'Skipping {full_filename}')


        


    @staticmethod
    def extract_performances(snapshot_path):
        jsonfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*/performance/*_metrics.json', recursive=True)]))
        jsonnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, jsonfiles))

        final_perf = []

        for idx, j in enumerate(jsonfiles):
            fi = open(j)
            jfile = json.load(fi)
            final_perf.append(jfile)


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
            
        performance.save(f'{snapshot_path}/performance_pytorch.xlsx')

        print(f'{snapshot_path}/performance_pytorch.xlsx' + ' generated')

    @staticmethod
    def extract_reject_performances(snapshot_path):
        jsonfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*/performance/*_metrics_reject.json', recursive=True)]))
        jsonnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, jsonfiles))

        final_perf = []

        for idx, j in enumerate(jsonfiles):
            fi = open(j)
            jfile = json.load(fi)
            final_perf.append(jfile)


        performance = openpyxl.Workbook()
        performance_ws = performance.active
        performance_ws.title = 'HAM10000'
        performance.create_sheet('ISIC2016')
        performance.create_sheet('ISIC2017')
        performance.create_sheet('ISIC2018')
        performance.create_sheet('KaggleMB')
        performance.create_sheet('7pointcriteria')


        cols = ['Network', 'DB Comb', 'Precision', 'Specificity', 'Sensitivity', 'Accuracy']

        HAM10000_ws = performance['HAM10000']
        HAM10000_ws.append(cols)

        

        for p in final_perf:
            HAM10000_ws.append([p['classifier'], str(p['dataset']), p['HAM10000']['Non-rejected']['precision'], 
            p['HAM10000']['Non-rejected']['specificity'], 
            p['HAM10000']['Non-rejected']['sensitivity'] if isinstance(p['HAM10000']['Non-rejected']['sensitivity'], (int, float)) else 'N/A', 
            p['HAM10000']['Non-rejected']['accuracy']])
            
        ISIC2016_ws = performance['ISIC2016']
        ISIC2016_ws.append(cols)

        for p in final_perf:
            ISIC2016_ws.append([p['classifier'], str(p['dataset']), p['ISIC2016']['Non-rejected']['precision'],
            p['ISIC2016']['Non-rejected']['specificity'], 
            p['ISIC2016']['Non-rejected']['sensitivity'] if isinstance(p['ISIC2016']['Non-rejected']['sensitivity'], (int, float)) else 'N/A', 
            p['ISIC2016']['Non-rejected']['accuracy']])

        ISIC2017_ws = performance['ISIC2017']
        ISIC2017_ws.append(cols)

        for p in final_perf:
            ISIC2017_ws.append([p['classifier'], str(p['dataset']), p['ISIC2017']['Non-rejected']['precision'], 
            p['ISIC2017']['Non-rejected']['specificity'], 
            p['ISIC2017']['Non-rejected']['sensitivity'] if isinstance(p['ISIC2017']['Non-rejected']['sensitivity'], (int, float)) else 'N/A', 
            p['ISIC2017']['Non-rejected']['accuracy']])

        ISIC2018_ws = performance['ISIC2018']
        ISIC2018_ws.append(cols)

        for p in final_perf:
            ISIC2018_ws.append([p['classifier'], str(p['dataset']), p['ISIC2018']['Non-rejected']['precision'], 
            p['ISIC2018']['Non-rejected']['specificity'], 
            p['ISIC2018']['Non-rejected']['sensitivity'] if isinstance(p['ISIC2018']['Non-rejected']['sensitivity'], (int, float)) else 'N/A', 
            p['ISIC2018']['Non-rejected']['accuracy']])


        KaggleMB_ws = performance['KaggleMB']
        KaggleMB_ws.append(cols)

        for p in final_perf:
            KaggleMB_ws.append([p['classifier'], str(p['dataset']), p['KaggleMB']['Non-rejected']['precision'], 
            p['KaggleMB']['Non-rejected']['specificity'], 
            p['KaggleMB']['Non-rejected']['sensitivity'] if isinstance(p['KaggleMB']['Non-rejected']['sensitivity'], (int, float)) else 'N/A', 
            p['KaggleMB']['Non-rejected']['accuracy']])

        _7pointcriteria_ws = performance['7pointcriteria']
        _7pointcriteria_ws.append(cols)

        for p in final_perf:
            _7pointcriteria_ws.append([p['classifier'], str(p['dataset']), p['_7_point_criteria']['Non-rejected']['precision'], 
            p['_7_point_criteria']['Non-rejected']['specificity'], 
            p['_7_point_criteria']['Non-rejected']['sensitivity'] if isinstance(p['_7_point_criteria']['Non-rejected']['sensitivity'], (int, float)) else 'N/A', 
            p['_7_point_criteria']['Non-rejected']['accuracy']])
            
        performance.save(f'{snapshot_path}/performance_reject.xlsx')

        print(f'{snapshot_path}/performance_reject.xlsx' + ' generated')
        

    def reject_uncertainties(snapshot_path, threshold=0.5):
        jsonfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*/performance/*_*_metrics.json', recursive=True)]))
        jsonnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, jsonfiles))
        jsonpaths = list(map(lambda x: pathlib.Path(os.path.dirname(x)), jsonfiles))

        

        plt.ioff()

        DBtypes = [db.name for db in mel.DatasetType]

        for idx, j in enumerate(jsonfiles):
            print(f'Rejecting {j} model')
            fi = open(j)
            jfile = json.load(fi)


            if not os.path.exists(os.path.join(jsonpaths[idx], 'plots')):
                os.makedirs(os.path.join(jsonpaths[idx], 'plots'), exist_ok=True)

            final_uncertainty = {}


            classifier_name = pathlib.Path(jsonpaths[idx]).parent.stem
            used_DB_list = [each_model for each_model in DBtypes if(each_model in jsonnames[idx])]
            DBnames = '+'.join(used_DB_list)
            
            final_uncertainty['dataset'] = used_DB_list
            final_uncertainty['classifier'] = classifier_name
            for db in ('HAM10000', 'ISIC2016', 'ISIC2017', 'ISIC2018', 'KaggleMB', '_7_point_criteria'):
                print(f"Rejecting {db} database")
                scores_all = np.array(jfile[db]['y_scores'])
                labels_all = np.array(jfile[db]['y_labels'])
                preds_all = np.array(jfile[db]['y_pred'])
                ids_all = np.array(jfile[db]['y_ids'])

                # Instantiate Rejector
                rej = ClassificationRejector(labels_all, scores_all)
                # Get entropy (Uncertainty Total (Entropy))
                # aleatoric - mean of TU (entropy), epistemic: TU - aleatoric
                total_unc = rej.uncertainty(unc_type="TU")
                all_unc = rej.uncertainty(unc_type=None)
                # Plotting uncertainty per test sample
                uncertainty_plot = rej.plot_uncertainty(unc_type="TU")
                
                # uncertainty_plot.suptitle(f'Testset: {db} \nTrainset: {DBnames} \nClassifier: {classifier_name}', fontsize=10, y=1.1, ha='left')
                uncertainty_plot.savefig(os.path.join(jsonpaths[idx], 'plots', DBnames+'_'+classifier_name+'_'+db+'_uncertainty'), bbox_inches='tight')
                # rej.plot_uncertainty(unc_type=None)
                # implement single rejection point
                reject_output = rej.reject(threshold=threshold, unc_type="TU", relative=True, show=True)

                cm = reject_cm(correct=rej.correct, unc_ary=total_unc, threshold= threshold,relative= True, show=False)
                
                reject_info = {}
                # False: Not rejected, True: rejected (removed)
                reject_info['y_labels'] = labels_all.tolist()
                reject_info['y_preds'] = preds_all.tolist()
                reject_info['y_scores'] = scores_all.tolist()
                reject_info['uncertainty'] = total_unc.tolist()
                reject_info['Threshold'] = threshold
                reject_info['Non-rejected'] = {}
                reject_info['Non-rejected']['y_labels'] = labels_all[cm[1] == False].tolist()
                reject_info['Non-rejected']['y_preds'] = preds_all[cm[1] == False].tolist()
                reject_info['Non-rejected']['y_scores'] = scores_all[cm[1] == False].tolist()
                reject_info['Non-rejected']['y_ids'] = ids_all[cm[1] == False].tolist()
                reject_info['Non-rejected']['Correct'] = cm[0][1]
                reject_info['Non-rejected']['Incorrect'] = cm[0][3]
                reject_info['Rejected'] = {}
                reject_info['Rejected']['y_labels'] = labels_all[cm[1] == True].tolist()
                reject_info['Rejected']['y_preds'] = preds_all[cm[1] == True].tolist()
                reject_info['Rejected']['y_scores'] = scores_all[cm[1] == True].tolist()
                reject_info['Rejected']['y_ids'] = ids_all[cm[1] == True].tolist()
                reject_info['Rejected']['Correct'] = {}
                reject_info['Rejected']['Correct']['total'] = cm[0][0]
                reject_info['Rejected']['Correct']['benign'] = \
                    int(np.sum((np.array(reject_info['Rejected']['y_labels']) == 0) & (np.array(reject_info['Rejected']['y_preds']) == 0)))
                reject_info['Rejected']['Correct']['malignant'] = \
                    int(np.sum((np.array(reject_info['Rejected']['y_labels']) == 1) & (np.array(reject_info['Rejected']['y_preds']) == 1)))
                reject_info['Rejected']['Incorrect'] = {}
                reject_info['Rejected']['Incorrect']['total'] = cm[0][2]
                reject_info['Rejected']['Incorrect']['benign'] = \
                    int(np.sum((np.array(reject_info['Rejected']['y_labels']) == 0) & (np.array(reject_info['Rejected']['y_preds']) == 1)))
                reject_info['Rejected']['Incorrect']['malignant'] = \
                    int(np.sum((np.array(reject_info['Rejected']['y_labels']) == 1) & (np.array(reject_info['Rejected']['y_preds']) == 0)))
                
                reject_info['Non-rejected_accuracy'] = reject_output[0][0]
                reject_info['Classification_quality'] = reject_output[0][1]
                reject_info['Rejection_quality'] = reject_output[0][2]
                
                # Relative threshold
                threshold_plt = rej.plot_reject(unc_type="TU", metric="NRA")
                threshold_plt.savefig(os.path.join(jsonpaths[idx], 'plots', DBnames+'_'+classifier_name+'_'+db+'_threshold'), bbox_inches='tight')
                # Absolute threshold
                # rej.plot_reject(unc_type="TU", metric="NRA", relative=False)

                common_binary_label = {
                        0.0: 'benign',
                }
                if (len(np.unique(reject_info['Non-rejected']['y_labels'])) == 2):
                    test_report = classification_report(reject_info['Non-rejected']['y_labels'], reject_info['Non-rejected']['y_preds'],
                    target_names=mel.Parser.common_binary_label.values(), output_dict = True)

                    reject_info['Non-rejected']['accuracy'] = test_report['accuracy']
                    reject_info['Non-rejected']['precision'] = test_report['macro avg']['precision']
                    reject_info['Non-rejected']['sensitivity'] = test_report['malignant']['recall']
                    reject_info['Non-rejected']['specificity'] = test_report['benign']['recall']
                    reject_info['Non-rejected']['f1-score'] = test_report['macro avg']['f1-score']

                elif (len(np.unique(reject_info['Non-rejected']['y_labels'])) == 1):
                    test_report = classification_report(reject_info['Non-rejected_y_labels'], reject_info['Non-rejected_y_preds'],
                    target_names=common_binary_label.values(), output_dict = True)

                    reject_info['Non-rejected']['accuracy'] = test_report['accuracy']
                    reject_info['Non-rejected']['precision'] = test_report['macro avg']['precision']
                    reject_info['Non-rejected']['sensitivity'] = '-'
                    reject_info['Non-rejected']['specificity'] = test_report['benign']['recall']
                    reject_info['Non-rejected']['f1-score'] = test_report['macro avg']['f1-score']

                # final_uncertainty.append(reject_info)

                final_uncertainty[db] = reject_info


            # Dump Json
            json_filename = f'{jsonnames[idx]}_reject.json'

            file_json = open(os.path.join(jsonpaths[idx], json_filename), "w")
            json.dump(final_uncertainty, file_json, indent = 6)
            file_json.close()

        # y_pred_all, y_true_all = generate_synthetic_output(NUM_SAMPLES, NUM_OBSERVATIONS)
        
        # (num_observations{IN+OOD}, num_samples, NUM_CLASSES)



	
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
    def ensemble(snapshot_path):
        jsonfiles = list(itertools.chain.from_iterable([glob.glob(f'{snapshot_path}/*/performance/*_metrics.json', recursive=True)]))
        jsonnames = list(map(lambda x: pathlib.Path(os.path.basename(x)).stem, jsonfiles))

        performances = []


        
        test_sets = ['HAM10000', 'ISIC2016', 'ISIC2017', 'ISIC2018', '_7_point_criteria', 'KaggleMB']
        ensemble = {}
        
            

        for idx, j in enumerate(jsonfiles):
            fi = open(j)
            jfile = json.load(fi)
            
            if idx == 0:
                for db in test_sets:
                    ensemble[db] = {}
                    ensemble[db]['y_labels'] = jfile[db]['y_labels']
            ensemble[idx] = {}
            ensemble[idx]['HAM10000'] = {}
            ensemble[idx]['HAM10000']['y_pred'] = jfile['HAM10000']['y_pred']
            ensemble[idx]['HAM10000']['y_scores'] = jfile['HAM10000']['y_scores']
            ensemble[idx]['ISIC2016'] = {}
            ensemble[idx]['ISIC2016']['y_pred'] = jfile['ISIC2016']['y_pred']
            ensemble[idx]['ISIC2016']['y_scores'] = jfile['ISIC2016']['y_scores']
            ensemble[idx]['ISIC2017'] = {}
            ensemble[idx]['ISIC2017']['y_pred'] = jfile['ISIC2017']['y_pred']
            ensemble[idx]['ISIC2017']['y_scores'] = jfile['ISIC2017']['y_scores']
            ensemble[idx]['ISIC2018'] = {}
            ensemble[idx]['ISIC2018']['y_pred'] = jfile['ISIC2018']['y_pred']
            ensemble[idx]['ISIC2018']['y_scores'] = jfile['ISIC2018']['y_scores']
            ensemble[idx]['_7_point_criteria'] = {}
            ensemble[idx]['_7_point_criteria']['y_pred'] = jfile['_7_point_criteria']['y_pred']
            ensemble[idx]['_7_point_criteria']['y_scores'] = jfile['_7_point_criteria']['y_scores']
            ensemble[idx]['KaggleMB'] = {}
            ensemble[idx]['KaggleMB']['y_pred'] = jfile['KaggleMB']['y_pred']
            ensemble[idx]['KaggleMB']['y_scores'] = jfile['KaggleMB']['y_scores']
            

        # ensemble['hard'] = {}
        # ensemble['soft'] = {}
        for db in test_sets:
            ensemble[db]['counts'] = [0] * len(ensemble[0][db]['y_pred'])
            # ensemble[db]['probs'] = [0] * len(ensemble[0][db]['y_scores'])
            ensemble[db]['probs'] = [0] * len(ensemble[0][db]['y_scores'])

        for idx in range(len(ensemble) - len(test_sets)):
            y_pred_HAM10000 = ensemble[idx]['HAM10000']['y_pred']
            y_pred_ISIC2016 = ensemble[idx]['ISIC2016']['y_pred']
            y_pred_ISIC2017 = ensemble[idx]['ISIC2017']['y_pred']
            y_pred_ISIC2018 = ensemble[idx]['ISIC2018']['y_pred']
            y_pred_KaggleMB = ensemble[idx]['KaggleMB']['y_pred']
            y_pred_7criteria = ensemble[idx]['_7_point_criteria']['y_pred']
            
            y_score_HAM10000 = ensemble[idx]['HAM10000']['y_scores']
            y_score_ISIC2016 = ensemble[idx]['ISIC2016']['y_scores']
            y_score_ISIC2017 = ensemble[idx]['ISIC2017']['y_scores']
            y_score_ISIC2018 = ensemble[idx]['ISIC2018']['y_scores']
            y_score_KaggleMB = ensemble[idx]['KaggleMB']['y_scores']
            y_score_7criteria = ensemble[idx]['_7_point_criteria']['y_scores']

            ensemble['HAM10000']['counts'] = [count + pred for count, pred in zip(ensemble['HAM10000']['counts'], y_pred_HAM10000)]
            ensemble['ISIC2016']['counts'] = [count + pred for count, pred in zip(ensemble['ISIC2016']['counts'], y_pred_ISIC2016)]
            ensemble['ISIC2017']['counts'] = [count + pred for count, pred in zip(ensemble['ISIC2017']['counts'], y_pred_ISIC2017)]
            ensemble['ISIC2018']['counts'] = [count + pred for count, pred in zip(ensemble['ISIC2018']['counts'], y_pred_ISIC2018)]
            ensemble['KaggleMB']['counts'] = [count + pred for count, pred in zip(ensemble['KaggleMB']['counts'], y_pred_KaggleMB)]
            ensemble['_7_point_criteria']['counts'] = [count + pred for count, pred in zip(ensemble['_7_point_criteria']['counts'], y_pred_7criteria)]
            

            ensemble['HAM10000']['probs'] = [score + pred for score, pred in zip(ensemble['HAM10000']['probs'], y_score_HAM10000)]
            ensemble['ISIC2016']['probs'] = [score + pred for score, pred in zip(ensemble['ISIC2016']['probs'], y_score_ISIC2016)]
            ensemble['ISIC2017']['probs'] = [score + pred for score, pred in zip(ensemble['ISIC2017']['probs'], y_score_ISIC2017)]
            ensemble['ISIC2018']['probs'] = [score + pred for score, pred in zip(ensemble['ISIC2018']['probs'], y_score_ISIC2018)]
            ensemble['KaggleMB']['probs'] = [score + pred for score, pred in zip(ensemble['KaggleMB']['probs'], y_score_KaggleMB)]
            ensemble['_7_point_criteria']['probs'] = [score + pred for score, pred in zip(ensemble['_7_point_criteria']['probs'], y_score_7criteria)]

        ensemble['HAM10000']['hard'] = [1 if count > len(ensemble['HAM10000']['counts']) / 2 else 0 for count in ensemble['HAM10000']['counts']]
        ensemble['HAM10000']['soft'] = [1 if score/ len(ensemble['HAM10000']['probs']) > 0.5 else 0 for score in ensemble['HAM10000']['probs']]


        print('stop')



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