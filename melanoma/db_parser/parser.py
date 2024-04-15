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
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import img_to_array, load_img, array_to_img


import melanoma as mel

class Parser:
    
    def __init__(self, base_dir, square_size, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype=None, uniform_normalization=True):
        self.base_dir = pathlib.Path(base_dir)

        self.logger = logging.getLogger('Melanoma classification')
        self.logger.setLevel(logging.DEBUG)
        self.square_size = square_size
        self.pseudo_num = pseudo_num
        self.resize_width = image_resize[1]
        self.resize_height = image_resize[0]
        self.split_ratio = split_ratio
        self.uniform_normalization = uniform_normalization

        self.classes_melanoma_binary = ['Non-Melanoma', 'Melanoma']

        if uniform_normalization is False:
            networkname = networktype.name
        elif uniform_normalization is True:
            networkname = 'uniform01'

        self.path = str(self.base_dir) + '/melanomaDB' + '/customDB' + '/' + networkname
		# data_gen_HAM10000, HAM10000_multiclass, HAM10000_binaryclass, data_gen_ISIC2016, ISIC2016_binaryclass = self.load(mode)
        isExist = os.path.exists(self.path)
        if not isExist :
            os.makedirs(self.path)
        else:
            pass

        print("path: ", self.base_dir)
		
		# Dataset path define
		
        now = datetime.now() # current date and time

        self.date_time = now.strftime("%m_%d_%Y_%H:%M")

        self.preprocessor = mel.Preprocess()

    def encode(self, img_array):
		
		
        img_pil = array_to_img(img_array)
        image_buffer = io.BytesIO() # convert array to bytes
        img_pil.save(image_buffer, format="JPEG", quality=100, subsampling=0)
        image_bytes = image_buffer.getvalue() # retrieve bytes string
        image_np = np.asarray(image_bytes)
        image_buffer.close()

        return image_np
	
    def decode(self, h5_file):
        
        
        pil_to_bytes_h5 = h5py.File(name=os.path.join(h5_file), mode='r', track_order=True)

        # print(pil_to_bytes_h5.keys())
        trainimages_key = np.array(pil_to_bytes_h5.get('trainimages'))
        testimages_key = np.array(pil_to_bytes_h5.get('testimages'))
        validationimages_key = np.array(pil_to_bytes_h5.get('validationimages'))
        trainlabels_key = np.array(pil_to_bytes_h5.get('trainlabels'))
        testlabels_key = np.array(pil_to_bytes_h5.get('testlabels'))
        validationlabels_key = np.array(pil_to_bytes_h5.get('validationlabels'))
        trainids_key = np.array(pil_to_bytes_h5.get('trainids'))
        testids_key = np.array(pil_to_bytes_h5.get('testids'))
        validationids_key = np.array(pil_to_bytes_h5.get('validationids'))

        for key in trainimages_key:
            image_array = np.array(pil_to_bytes_h5['trainimages'][key][()])
            image_ids = np.array(pil_to_bytes_h5['trainids'][key][()])
            
            image_buffer = io.BytesIO(image_array)
            image_pil = Image.open(image_buffer)

            # process the pillow image image_pil ...

        pil_to_bytes_h5.close()
        
        
        return image_pil

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
            img_np = self.encode(img)
            trainimages_grp.create_dataset(f'{trainids[idx]}_{idx}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=img_np)
        for idx, img in enumerate(testpxs):
            img_np = self.encode(img)
            testimages_grp.create_dataset(f'{testids[idx]}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=img_np)
        for idx, img in enumerate(validationpxs):
            img_np = self.encode(img)
            validationimages_grp.create_dataset(f'{validationids[idx]}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=img_np)

        for idx, label in enumerate(trainlabels):
            trainlabels_grp.create_dataset(f'{trainids[idx]}_{idx}', track_order=True, shape=(2, ), maxshape=(None, ), compression='gzip', data=label)
        for idx, label in enumerate(testlabels):
            testlabels_grp.create_dataset(f'{testids[idx]}', track_order=True, shape=(2, ), maxshape=(None, ), compression='gzip', data=label)
        for idx, label in enumerate(validationlabels):
            validationlabels_grp.create_dataset(f'{validationids[idx]}', track_order=True, shape=(2, ), maxshape=(None, ), compression='gzip', data=label)

        for idx, id in enumerate(trainids):
            trainids_grp.create_dataset(f'{trainids[idx]}_{idx}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=id)
        for idx, id in enumerate(testids):
            testids_grp.create_dataset(f'{testids[idx]}', shape=(1, ), track_order=True, maxshape=(None, ), compression='gzip', data=id)
        for idx, id in enumerate(validationids):
            validationids_grp.create_dataset(f'{validationids[idx]}', track_order=True, shape=(1, ), maxshape=(None, ), compression='gzip', data=id)

        hf.close()

    def makeFolders(self, datasetname):
        
        
        debug_rgb_folder = self.path + f'/debug/{datasetname}/RGB_resized/'+f'{self.resize_height}h_{self.resize_width}w_{self.date_time}'
        debug_feature_folder = self.path + f'/debug/{datasetname}/feature/'+f'{self.resize_height}h_{self.resize_width}w_{self.date_time}'
        debugRgbFolderExist = os.path.exists(debug_rgb_folder)
        debugFeatureFolderExist = os.path.exists(debug_feature_folder)
        if not debugRgbFolderExist :
            os.makedirs(debug_rgb_folder)
        else:
            pass
        if not debugFeatureFolderExist :
            os.makedirs(debug_feature_folder)
        else:
            pass
        self.whole_rgb_folder = debug_rgb_folder + '/Whole'
        self.train_rgb_folder = debug_rgb_folder + '/Train'
        self.val_rgb_folder = debug_rgb_folder + '/Val'
        self.test_rgb_folder = debug_rgb_folder + '/Test'

        self.whole_feature_folder = debug_feature_folder + '/Whole'
        self.train_feature_folder = debug_feature_folder + '/Train'
        self.val_feature_folder = debug_feature_folder + '/Val'
        self.test_feature_folder = debug_feature_folder + '/Test'



        self.isWholeRGBExist = os.path.exists(self.whole_rgb_folder)
        self.isTrainRGBExist = os.path.exists(self.train_rgb_folder)
        self.isValRGBExist = os.path.exists(self.val_rgb_folder)
        self.isTestRGBExist = os.path.exists(self.test_rgb_folder)
        self.isWholeFeatureExist = os.path.exists(self.whole_feature_folder)
        self.isTrainFeatureExist = os.path.exists(self.train_feature_folder)
        self.isValFeatureExist = os.path.exists(self.val_feature_folder)
        self.isTestFeatureExist = os.path.exists(self.test_feature_folder)

        pd.set_option('display.max_columns', 500)

        # Given lesion types


        # Not required for pickled data
        # resize() order: (width, height)
        
        

    def saveDatasetToFile(self, augment_ratio=None):
        pass