import os
import io
import pathlib
from glob import glob
import logging
from IPython.display import display
from PIL import Image
from datetime import datetime
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
# from keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.models import load_model


import melanoma as mel

class Parser:
    
    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        # self.base_dir = pathlib.Path(base_dir)

        self.base_dir = base_dir
        self.logger = logging.getLogger('Melanoma classification')
        self.logger.setLevel(logging.DEBUG)
        self.pseudo_num = pseudo_num
        self.split_ratio = split_ratio
        

        self.classes_melanoma_binary = ['benign', 'malignant']

        

        self.common_binary_label = {
			0.0: 'benign',
			1.0: 'malignant',
		}

        
		
		# Dataset path define
		
        

        # self.preprocessor = mel.Preprocess()

    @staticmethod
    def encode(img_pil):
		
		
        # img_pil = array_to_img(img_array)
        image_buffer = io.BytesIO() # convert array to bytes
        # img_pil.save(image_buffer, format="JPEG", quality=100, subsampling=0)
        img_pil.save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue() # retrieve bytes string
        image_np = np.asarray(image_bytes, dtype=f'|S{len(image_bytes)}')
        image_buffer.close()

        return image_np
    
    @staticmethod
    def decode(np_bytes):
        
        image_buffer = io.BytesIO(np_bytes)
        image_pil = Image.open(image_buffer)        
        
        return image_pil
	
    @staticmethod
    def open_H5(h5file):
        
        hf = h5py.File(name=h5file, mode='r', track_order=True)
        
        trainimages_key = np.array(hf.get('trainimages'))
        trainlabels_key = np.array(hf.get('trainlabels'))
        trainids_key = np.array(hf.get('trainids'))
        validationimages_key = np.array(hf.get('validationimages'))
        validationlabels_key = np.array(hf.get('validationlabels'))
        validationids_key = np.array(hf.get('validationids'))
        testimages_key = np.array(hf.get('testimages'))
        testlabels_key = np.array(hf.get('testlabels'))
        testids_key = np.array(hf.get('testids'))

        assert (trainimages_key == trainlabels_key).all() == True
        assert (trainimages_key == trainids_key).all() == True
        assert (validationimages_key == validationlabels_key).all() == True
        assert (validationimages_key == validationids_key).all() == True
        assert (testimages_key == testlabels_key).all() == True
        assert (testimages_key == testids_key).all() == True

        trainimages_list = []
        trainlabels_list = []
        trainids_list = []
        validationimages_list = []
        validationlabels_list = []
        validationids_list = []
        testimages_list = []
        testlabels_list = []
        testids_list = []

        for key in trainimages_key:
            trainimages_list.append(np.array(hf.get('trainimages')[key]))
            trainlabels_list.append(np.array(hf.get('trainlabels')[key]))
            trainids_list.append(np.array(hf.get('trainids')[key]))

        for key in validationimages_key:
            validationimages_list.append(np.array(hf.get('validationimages')[key]))
            validationlabels_list.append(np.array(hf.get('validationlabels')[key]))
            validationids_list.append(np.array(hf.get('validationids')[key]))

        for key in testimages_key:
            testimages_list.append(np.array(hf.get('testimages')[key]))
            testlabels_list.append(np.array(hf.get('testlabels')[key]))
            testids_list.append(np.array(hf.get('testids')[key]))
        
        traindata = {"trainimages": trainimages_list,
                     "trainlabels": trainlabels_list,
                     "trainids": trainids_list
                    }
        
        validationdata = {"validationimages": validationimages_list,
                     "validationlabels": validationlabels_list,
                     "validationids": validationids_list
                    }
        
        testdata = {"testimages": testimages_list,
                     "testlabels": testlabels_list,
                     "testids": testids_list
                    }


        

        hf.close()
        
        return traindata, validationdata, testdata
        
        

    def validate_h5(self, path, filename, dbnumimgs, train_only=True, val_exists=False, test_exists=False):
        
        
        hf = h5py.File(name=os.path.join(path, filename), mode='r', track_order=True)
        
        # print(pil_to_bytes_h5.keys())
        trainimages_key = np.array(hf.get('trainimages'))
        trainlabels_key = np.array(hf.get('trainlabels'))
        trainids_key = np.array(hf.get('trainids'))
        validationimages_key = np.array(hf.get('validationimages'))
        validationlabels_key = np.array(hf.get('validationlabels'))
        validationids_key = np.array(hf.get('validationids'))
        testimages_key = np.array(hf.get('testimages'))
        testlabels_key = np.array(hf.get('testlabels'))
        testids_key = np.array(hf.get('testids'))
        

        if train_only is True:
            assert len(trainimages_key)+len(validationimages_key)+len(testimages_key) == dbnumimgs['trainimages']
            assert len(trainlabels_key)+len(validationlabels_key)+len(testlabels_key) == dbnumimgs['trainimages']
            assert len(trainids_key)+len(validationids_key)+len(testids_key) == dbnumimgs['trainimages']

        elif train_only is False and val_exists is True:
            
            assert len(trainimages_key) + len(validationimages_key) == dbnumimgs['trainimages']+dbnumimgs['validationimages']
            assert len(trainlabels_key) + len(validationlabels_key) == dbnumimgs['trainimages'] + dbnumimgs['validationimages']
            assert len(trainids_key) + len(validationids_key) == dbnumimgs['trainimages'] + dbnumimgs['validationimages']

        if train_only is False and test_exists is True:

            assert len(testimages_key)+len(validationimages_key)+len(trainimages_key) == dbnumimgs['testimages']+dbnumimgs['validationimages']+dbnumimgs['trainimages']
            assert len(testlabels_key)+len(validationlabels_key)+len(trainlabels_key) == dbnumimgs['testimages']+dbnumimgs['validationimages']+dbnumimgs['trainimages']
            assert len(testids_key)+len(validationids_key)+len(trainids_key) == dbnumimgs['testimages']+dbnumimgs['validationimages']+dbnumimgs['trainimages']


        hf.close()
        
        
        

    def generateHDF5(self, path, filename, trainpxs, testpxs, validationpxs,
                     trainids, testids, validationids,
                     trainlabels, testlabels, validationlabels):
        hf = h5py.File(name=os.path.join(path, filename), mode='w', track_order=True)
        # with h5py.File(os.path.join(path, filename), 'w') as hf:
        trainimages_grp = hf.create_group(name='trainimages', track_order=True)
        testimages_grp = hf.create_group(name='testimages', track_order=True)
        validationimages_grp = hf.create_group(name='validationimages', track_order=True)
        trainlabels_grp = hf.create_group(name='trainlabels', track_order=True)
        testlabels_grp = hf.create_group(name='testlabels', track_order=True)
        validationlabels_grp = hf.create_group(name='validationlabels', track_order=True)
        trainids_grp = hf.create_group(name='trainids', track_order=True)
        testids_grp = hf.create_group(name='testids', track_order=True)
        validationids_grp = hf.create_group(name='validationids', track_order=True)
        for idx, img in enumerate(trainpxs):
            # img_np = self.encode(img)
            trainimages_grp.create_dataset(f'{trainids[idx]}_{idx}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=img)
        for idx, img in enumerate(testpxs):
            # img_np = self.encode(img)
            testimages_grp.create_dataset(f'{testids[idx]}_{idx}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=img)
        for idx, img in enumerate(validationpxs):
            # img_np = self.encode(img)
            validationimages_grp.create_dataset(f'{validationids[idx]}_{idx}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=img)

        for idx, label in enumerate(trainlabels):
            trainlabels_grp.create_dataset(f'{trainids[idx]}_{idx}', track_order=True, shape=(2, ), maxshape=(None, ), compression='gzip', data=label)
        for idx, label in enumerate(testlabels):
            testlabels_grp.create_dataset(f'{testids[idx]}_{idx}', track_order=True, shape=(2, ), maxshape=(None, ), compression='gzip', data=label)
        for idx, label in enumerate(validationlabels):
            validationlabels_grp.create_dataset(f'{validationids[idx]}_{idx}', track_order=True, shape=(2, ), maxshape=(None, ), compression='gzip', data=label)

        for idx, id in enumerate(trainids):
            trainids_grp.create_dataset(f'{trainids[idx]}_{idx}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=id)
        for idx, id in enumerate(testids):
            testids_grp.create_dataset(f'{testids[idx]}_{idx}', shape=(1, ), track_order=True, maxshape=(None, ), compression='gzip', data=id)
        for idx, id in enumerate(validationids):
            validationids_grp.create_dataset(f'{validationids[idx]}_{idx}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=id)

        hf.close()

        



    def makeFolders(self, datasetname):
        
        path = os.path.join(self.base_dir, 'data', 'melanomaDB', datasetname)
		# data_gen_HAM10000, HAM10000_multiclass, HAM10000_binaryclass, data_gen_ISIC2016, ISIC2016_binaryclass = self.load(mode)
        isExist = os.path.exists(path)
        if not isExist :
            os.makedirs(path)
        else:
            pass

        print("path: ", path)

        now = datetime.now() # current date and time

        date_time = now.strftime("%m_%d_%Y_%H:%M")
        
        rgb_folder = os.path.join(path, date_time)
        rgbFolderExist = os.path.exists(rgb_folder)
        
        if not rgbFolderExist :
            os.makedirs(rgb_folder)
        else:
            pass
        
        self.whole_rgb_folder = rgb_folder + '/Whole'
        self.train_rgb_folder = rgb_folder + '/Train'
        self.val_rgb_folder = rgb_folder + '/Val'
        self.test_rgb_folder = rgb_folder + '/Test'



        self.isWholeRGBExist = os.path.exists(self.whole_rgb_folder)
        self.isTrainRGBExist = os.path.exists(self.train_rgb_folder)
        self.isValRGBExist = os.path.exists(self.val_rgb_folder)
        self.isTestRGBExist = os.path.exists(self.test_rgb_folder)
        
        

    def saveDatasetToFile(self):
        pass