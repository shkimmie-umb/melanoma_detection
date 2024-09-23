import tensorflow as tf
from .model import Model as Base_Model
# from .util import DatasetType
from melanoma import augmentationStrategy as aug
# from .util import NetworkType
import melanoma as mel
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
import random
import math
import os
import pathlib
from tqdm import tqdm
from PIL import Image
from enum import Enum
from glob import glob
import albumentations as A
import io

# from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


	

class Preprocess:

    def __init__(self):
        # self.square_size = square_size
        # desired size
        # if image_size is not None:
        #     self.img_width = image_size[1]
        #     self.img_height = image_size[0]
        pass

    def augmentation(self, train_rgb_folder, square_size, resize_width, resize_height, augment_ratio, df_trainset):
        def aug_logic(df_mel, df_non_mel, make_50_50=False):
            augMethod = aug.Augmentation(aug.crop_flip_brightnesscontrast())

            df_augmented_cls0 = pd.DataFrame(data=None, columns=df_trainset.columns)
            df_augmented_cls1 = pd.DataFrame(data=None, columns=df_trainset.columns)

            mel_cnt = df_mel.shape[0]
            non_mel_cnt = df_non_mel.shape[0]

            if mel_cnt > non_mel_cnt:
                larger_cls = df_mel
                smaller_cls = df_non_mel
            elif mel_cnt < non_mel_cnt:
                larger_cls = df_non_mel
                smaller_cls = df_mel

            def do_augmentation(each_df, df_augmented, df_trainset, square_size, resize_width, resize_height):
                random_idx = random.choice(each_df.index)
                assert each_df.path[random_idx] == df_trainset.path[random_idx]

                # img = self.squareImgs(path=df_trainset.path[random_idx], square_size=square_size)
                # Center crop & Resize
                img = load_img(path=df_trainset.path[random_idx], target_size=None)
                np_img = img_to_array(img=img, dtype='uint8')
                # df_mel_augmented.iloc[j] = df_trainset.loc[randmel_idx]
                df_augmented = pd.concat([df_augmented, df_trainset.loc[[random_idx]]], ignore_index=True)
                augmented_img = augMethod.augmentation(
                    input_img=np_img,
                    square_size=square_size,
                    crop_height=resize_height, 
                    crop_width=resize_width, 
                    zoomout=1.0, 
                    zoomin=1.2, 
                    p_scaling=0.5, 
                    p_rotation=0.5, 
                    p_hflip=0.5, 
                    p_vflip=0.5,
                    p_randomBrightnessContrast=0.5)
                pil_aug = mel.Parser.encode(array_to_img(augmented_img['image']))
                df_augmented.loc[df_augmented.index[-1], 'image'] = pil_aug

                return df_augmented
                
            # Whatever the class with less images, we augment to the # of other class X augment_ratio
            for i in range(smaller_cls.shape[0], math.ceil(larger_cls.shape[0] * augment_ratio)):
                df_augmented_cls0 = do_augmentation(
                                each_df=smaller_cls,
                                df_augmented=df_augmented_cls0,
                                df_trainset=df_trainset,
                                square_size=square_size,
                                resize_width=resize_width,
                                resize_height=resize_height)
            # Whatever the class with more images, we augment to the # of the class itself X augment_ratio
            for i in range(larger_cls.shape[0], math.ceil(larger_cls.shape[0] * augment_ratio)):
                df_augmented_cls1 = do_augmentation(
                                each_df=larger_cls,
                                df_augmented=df_augmented_cls1,
                                df_trainset=df_trainset,
                                square_size=square_size,
                                resize_width=resize_width,
                                resize_height=resize_height)

            df_augmented = pd.concat([df_augmented_cls0, df_augmented_cls1], ignore_index=True, axis=0)
            # df_mel_augmented = df_augmented[df_augmented.cell_type_binary == 'Melanoma']
            # df_non_mel_augmented = df_augmented[df_augmented.cell_type_binary == 'Non-Melanoma']

            return df_augmented


        df_mel = df_trainset[df_trainset.cell_type_binary=='Melanoma']
        df_non_mel = df_trainset[df_trainset.cell_type_binary=='Non-Melanoma']

        df_augmented = aug_logic(df_mel, df_non_mel)

        df_mel_augmented_cnt = df_augmented[df_augmented.cell_type_binary=='Melanoma'].shape[0]
        df_non_mel_augmented_cnt = df_augmented[df_augmented.cell_type_binary=='Non-Melanoma'].shape[0]

        # df_mel_augmented = pd.DataFrame(columns=df_trainset.columns.tolist())
        # df_non_mel_augmented = pd.DataFrame(columns=df_trainset.columns.tolist())
        
        

        labels = df_trainset['cell_type_binary'].unique()

        augmentation_folder = f"{train_rgb_folder}/augmented"
        isAugFolderExist = os.path.exists(augmentation_folder)
        if not isAugFolderExist:
            for i in labels:
                os.makedirs(f"{augmentation_folder}/{i}", exist_ok=True)

        # Save augmented images for viewing purpose
        for idx in df_augmented.index:
            # img = Image.fromarray(df_mel_augmented.image[idx], mode='RGB')
            img = mel.Parser.decode(df_augmented['image'][idx])
            currentPath = pathlib.Path(df_augmented.path[idx])
            label = df_augmented.cell_type_binary[idx]
            img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg")

        trainimages = list(map(lambda x:x[0], df_trainset['image'])) # Filter out only pixel from the list
        trainimages_augmented_only = list(map(lambda x:x, df_augmented['image'])) # Filter out only pixel from the list

        
        # imgs_augmented = self.normalizeImgs(imgs=trainpixels_augmented, networktype=networktype, uniform_normalization=uniform_normalization)

        # trainimages_augmented = np.vstack((trainimages, trainpixels_augmented))
        # trainimages_augmented = pd.concat([trainimages, trainpixels_augmented], ignore_index=True, axis=0)
        trainimages_augmented = trainimages + trainimages_augmented_only
        
        # ids augmented
        trainids = list(map(lambda x:pathlib.Path(x).stem, df_trainset.path))
        trainids_augmented_only = list(map(lambda x:pathlib.Path(x).stem, df_augmented.path))
        trainids_augmented = trainids + trainids_augmented_only
        
        # labels_augmented = np.asarray(trainset_HAM10000_augmented.cell_type_binary_idx, dtype='float64')
        trainlabels = to_categorical(df_trainset.cell_type_binary_idx, num_classes=2)
        trainlabels_augmented_only = to_categorical(df_augmented.cell_type_binary_idx, num_classes=2)
        trainlabels_augmented = np.vstack((trainlabels, trainlabels_augmented_only))
        # trainlabels_augmented = trainlabels + trainlabels_augmented_only

        assert len(trainimages_augmented) == len(trainids_augmented)
        assert len(trainids_augmented) == len(trainlabels_augmented)

        # df_mel_augmented = df_augmented[df_augmented.cell_type_binary=='Melanoma']
        # df_non_mel_augmented = df_augmented[df_augmented.cell_type_binary=='Non-Melanoma']

        
        return df_mel_augmented_cnt, df_non_mel_augmented_cnt, trainimages_augmented, trainlabels_augmented, trainids_augmented
    


    @staticmethod
    def saveNumpyImagesToFiles(df, base_path):
        for idx, sliced_idx in enumerate(df.index):
            # Print will make the logic fail
            # print(df.image[sliced_idx][1])
            img = mel.Parser.decode(df['image'][sliced_idx][0])
            currentPath = pathlib.Path(df['image'][sliced_idx][1])
            label = df.cell_type_binary[sliced_idx]
            # assert label == original_df.cell_type_binary[sliced_idx]
            img.save(f"{base_path}/{label}/{currentPath.stem}.jpg", quality=95)

    
    # def saveNumpyImagesToFilesWithoutLabel(self, df, base_path):
    #     def assert_(cond): assert cond
    #     return df.index.map(lambda x: (
    #             img := load_img(path=df.image[x][1], target_size=None),
	# 			currentPath := pathlib.Path(df.image[x][1]), # [0]: Encoded PIL obj, [1]: PosixPath
	# 			# label := df.cell_type_binary[x],
	# 			# assert_(label == original_df.cell_type_binary[x]),
	# 			img.save(f"{base_path}/{currentPath}.jpg", quality=100, subsampling=0)
	# 		))

    def saveCustomDBImagesToFiles(self, labels, foldersExist, RGBfolders, trains, vals=None, tests=None):
        
        

        isWholeRGBExist = foldersExist['isWholeRGBExist']
        isTrainRGBExist = foldersExist['isTrainRGBExist']
        isValRGBExist = foldersExist['isValRGBExist']
        isTestRGBExist = foldersExist['isTestRGBExist']

        whole_rgb_folder = RGBfolders['whole_rgb_folder']
        train_rgb_folder = RGBfolders['train_rgb_folder']
        val_rgb_folder = RGBfolders['val_rgb_folder']
        test_rgb_folder = RGBfolders['test_rgb_folder']

        if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
            for i in labels:
                os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)

        path_benign_train = trains['path_benign_train']
        path_malignant_train = trains['path_malignant_train']
        path_dict_benign_train = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path_benign_train, '*.*'))}
        path_dict_malignant_train = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path_malignant_train, '*.*'))}

        if vals is not None:
            path_benign_val = vals['path_benign_val']
            path_malignant_val = vals['path_malignant_val']
            path_dict_benign_val = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path_malignant_val, '*.*'))}
            path_dict_malignant_val = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path_malignant_val, '*.*'))}

        if tests is not None:
            path_benign_test = tests['path_benign_test']
            path_malignant_test = tests['path_malignant_test']
            path_dict_malignant_train = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path_malignant_train, '*.*'))}
            path_dict_malignant_train = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path_malignant_train, '*.*'))}
        


        # map(lambda x: (
		# 		currentPath_train := pathlib.Path(df.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
		# 		label := df.cell_type_binary[x],
		# 		assert_(label == original_df.cell_type_binary[x]),
		# 		df.image[x][0].save(f"{base_path}/{label}/{currentPath_train.name}", quality=100, subsampling=0)
		# 	))