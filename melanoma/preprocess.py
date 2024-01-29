import tensorflow as tf
from .model import Model as Base_Model
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

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


	

class Preprocess:

    def __init__(self, image_size):
        self.img_width = image_size[1]
        self.img_height = image_size[0]
        

    def normalizeImgs_ResNet50(self, imgs):
        imgList = []
        for img in imgs:
            img = np.expand_dims(img, axis=0)
            transformed_img = tf.keras.applications.resnet.preprocess_input(img) # RGB to BGR
            # imgList.append(transformed_img/255.)
            imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def normalizeImgs_Xception(self, imgs):
        imgList = []
        for img in imgs:
            img = np.expand_dims(img, axis=0)
            transformed_img = tf.keras.applications.xception.preprocess_input(img) # RGB to BGR
            # imgList.append(transformed_img/255.)
            imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def normalizeImgs_inceptionV3(self, imgs):
        imgList = []
        for img in imgs:
            img = np.expand_dims(img, axis=0)
            transformed_img = tf.keras.applications.inception_v3.preprocess_input(img) # RGB to BGR
            # imgList.append(transformed_img/255.)
            imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def normalizeImgs_vgg16(self, imgs):
        imgList = []
        for img in imgs:
            img = np.expand_dims(img, axis=0)
            transformed_img = tf.keras.applications.vgg16.preprocess_input(img) # RGB to BGR
            # imgList.append(transformed_img/255.)
            imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def normalizeImgs_vgg19(self, imgs):
        imgList = []
        for img in imgs:
            img = np.expand_dims(img, axis=0)
            transformed_img = tf.keras.applications.vgg19.preprocess_input(img) # RGB to BGR
            # imgList.append(transformed_img/255.)
            imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def Dataset_loader_ResNet50(self, imgPath):
        imgList = []
        read = lambda imname: load_img(path=imname, target_size=(self.img_height, self.img_width))
        for idx, IMAGE_NAME in enumerate(tqdm(os.listdir(imgPath))):
            PATH = os.path.join(imgPath,IMAGE_NAME)
            _, ftype = os.path.splitext(PATH)
            if ftype == ".jpg":
                img = read(PATH)
                img = np.expand_dims(img, axis=0)
                transformed_img = tf.keras.applications.resnet.preprocess_input(img)
                # imgList.append(transformed_img/255.)
                imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def Dataset_loader_Xception(self, imgPath):
        imgList = []
        read = lambda imname: load_img(path=imname, target_size=(self.img_height, self.img_width))
        for idx, IMAGE_NAME in enumerate(tqdm(os.listdir(imgPath))):
            PATH = os.path.join(imgPath,IMAGE_NAME)
            _, ftype = os.path.splitext(PATH)
            if ftype == ".jpg":
                img = read(PATH)
                img = np.expand_dims(img, axis=0)
                transformed_img = tf.keras.applications.xception.preprocess_input(img)
                # imgList.append(transformed_img/255.)
                imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def Dataset_loader_inceptionV3(self, imgPath):
        imgList = []
        read = lambda imname: load_img(path=imname, target_size=(self.img_height, self.img_width))
        for idx, IMAGE_NAME in enumerate(tqdm(os.listdir(imgPath))):
            PATH = os.path.join(imgPath,IMAGE_NAME)
            _, ftype = os.path.splitext(PATH)
            if ftype == ".jpg":
                img = read(PATH)
                img = np.expand_dims(img, axis=0)
                transformed_img = tf.keras.applications.inception_v3.preprocess_input(img)
                # imgList.append(transformed_img/255.)
                imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def Dataset_loader_vgg16(self, imgPath):
        imgList = []
        read = lambda imname: load_img(path=imname, target_size=(self.img_height, self.img_width))
        for idx, IMAGE_NAME in enumerate(tqdm(os.listdir(imgPath))):
            PATH = os.path.join(imgPath,IMAGE_NAME)
            _, ftype = os.path.splitext(PATH)
            if ftype == ".jpg":
                img = read(PATH)
                img = np.expand_dims(img, axis=0)
                transformed_img = tf.keras.applications.vgg16.preprocess_input(img)
                # imgList.append(transformed_img/255.)
                imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def Dataset_loader_vgg19(self, imgPath):
        imgList = []
        read = lambda imname: load_img(path=imname, target_size=(self.img_height, self.img_width))
        for idx, IMAGE_NAME in enumerate(tqdm(os.listdir(imgPath))):
            PATH = os.path.join(imgPath,IMAGE_NAME)
            _, ftype = os.path.splitext(PATH)
            if ftype == ".jpg":
                img = read(PATH)
                img = np.expand_dims(img, axis=0)
                transformed_img = tf.keras.applications.vgg19.preprocess_input(img)
                # imgList.append(transformed_img/255.)
                imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG
    
    def augmentation(self, datasettype, networktype, train_rgb_folder, labels, trainimages, trainlabels, augment_ratio, df_trainset):
        
        # augMethod = aug.Augmentation(aug.crop_flip_brightnesscontrast())
        augMethod = aug.Augmentation(aug.crop_flip())

        resize_ratio = 0.75

        # heightsz_weight = -(1-0.75)*((img_height-min_height)/(max_height-min_height))+1
        # widthsz_weight = -(1-0.75)*((img_width-min_width)/(max_width-min_width))+1


        df_mel = df_trainset[df_trainset.cell_type_binary=='Melanoma']
        # df_mel = df_mel[df_mel.apply(lambda x:x.img_sizes[0] * -(1-0.75)*((x.img_sizes[0]-min_height)/(max_height-min_height))+1 > self.img_height, axis=1)]
        cnt_mel = df_mel.shape[0]
        # Filtering out small images less than cropping size
        df_mel = df_mel[df_mel.apply(lambda x: x.img_sizes[0] * resize_ratio > self.img_height, axis=1)]
        cnt_mel_filt = df_mel.shape[0]
        diff_mel = cnt_mel - cnt_mel_filt
        df_non_mel = df_trainset[df_trainset.cell_type_binary=='Non-Melanoma']
        cnt_non_mel = df_non_mel.shape[0]
        # Filtering out small images less than cropping size
        df_non_mel = df_non_mel[df_non_mel.apply(lambda x: x.img_sizes[1] * resize_ratio > self.img_width, axis=1)]
        cnt_non_mel_filt = df_non_mel.shape[0]
        diff_non_mel = cnt_non_mel - cnt_non_mel_filt

        mel_cnt = df_mel.shape[0]
        non_mel_cnt = df_non_mel.shape[0]

        df_mel_augmented = pd.DataFrame(columns=df_trainset.columns.tolist())
        df_non_mel_augmented = pd.DataFrame(columns=df_trainset.columns.tolist())
        

        if mel_cnt < non_mel_cnt:
            # melanoma augmentation here
            # Melanoma images will be augmented to the N times number of the Non-melanoma images
            for j, id in enumerate(range((non_mel_cnt - mel_cnt), math.ceil(non_mel_cnt * augment_ratio))):
                randmel_idx = random.choice(df_mel.index)
                assert df_mel.path[randmel_idx] == df_trainset.path[randmel_idx]
                # img = Image.open(df_trainset.path[randmel_idx]).convert("RGB")
                img = load_img(path=df_trainset.path[randmel_idx], target_size=None)
                np_img = img_to_array(img)
                df_mel_augmented.loc[j] = df_trainset.loc[randmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                # display(f'path: {df_trainset.path[randmel_idx]}')
                augmented_img = augMethod.augmentation(input_img=np_img, resize_ratio=resize_ratio, crop_height=self.img_height, crop_width=self.img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5)
                df_mel_augmented.at[j, 'image'] = None
                df_mel_augmented.at[j, 'image'] = augmented_img['image']
                num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - (non_mel_cnt - mel_cnt)
                assert df_mel_augmented.shape[0] <= num_augmented_img
                

            # non-melanoma augmentation here
            for j, id in enumerate(range(non_mel_cnt, math.ceil(non_mel_cnt * augment_ratio))):
                randnonmel_idx = random.choice(df_non_mel.index)
                assert df_non_mel.path[randnonmel_idx] == df_trainset.path[randnonmel_idx]
                # img = Image.open(df_trainset.path[randnonmel_idx]).convert("RGB")
                img = load_img(path=df_trainset.path[randnonmel_idx], target_size=None)
                np_img = img_to_array(img)
                df_non_mel_augmented.loc[j] = df_trainset.loc[randnonmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                augmented_img = augMethod.augmentation(input_img=np_img, resize_ratio=resize_ratio, crop_height=self.img_height, crop_width=self.img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5)
                df_non_mel_augmented.at[j, 'image'] = None
                df_non_mel_augmented.at[j, 'image'] = augmented_img['image']
                num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - non_mel_cnt
                assert df_non_mel_augmented.shape[0] <= num_augmented_img
        elif mel_cnt > non_mel_cnt:
            # melanoma augmentation here
            for j, id in enumerate(range(mel_cnt, math.ceil(mel_cnt * augment_ratio))):
                randmel_idx = random.choice(df_mel.index)
                assert df_mel.path[randmel_idx] == df_trainset.path[randmel_idx]
                # img = Image.open(df_trainset.path[randmel_idx]).convert("RGB")
                img = load_img(path=df_trainset.path[randmel_idx], target_size=None)
                np_img = img_to_array(img)
                df_mel_augmented.loc[j] = df_trainset.loc[randmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                augmented_img = augMethod.augmentation(input_img=np_img, resize_ratio=resize_ratio, crop_height=self.img_height, crop_width=self.img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5)
                df_mel_augmented.at[j, 'image'] = None
                df_mel_augmented.at[j, 'image'] = augmented_img['image']
                num_augmented_img = math.ceil(mel_cnt * augment_ratio) - mel_cnt
                assert df_mel_augmented.shape[0] <= num_augmented_img

            # non-melanoma augmentation here
            for j, id in enumerate(range((mel_cnt - non_mel_cnt), math.ceil(mel_cnt * augment_ratio))):
                randnonmel_idx = random.choice(df_non_mel.index)
                assert df_non_mel.path[randnonmel_idx] == df_trainset.path[randnonmel_idx]
                # img = Image.open(df_trainset.path[randnonmel_idx]).convert("RGB")
                img = load_img(path=df_trainset.path[randnonmel_idx], target_size=None)
                # np_img = np.asarray(img)
                np_img = img_to_array(img)
                df_non_mel_augmented.loc[j] = df_trainset.loc[randnonmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                augmented_img = augMethod.augmentation(input_img=np_img, resize_ratio=resize_ratio, crop_height=self.img_height, crop_width=self.img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5)
                df_non_mel_augmented.at[j, 'image'] = None
                df_non_mel_augmented.at[j, 'image'] = augmented_img['image']
                num_augmented_img = math.ceil(mel_cnt * augment_ratio) - (mel_cnt - non_mel_cnt)
                assert df_non_mel_augmented.shape[0] <= num_augmented_img

        df_trainset_augmented = pd.concat([df_mel_augmented, df_non_mel_augmented])

        augmentation_folder = f"{train_rgb_folder}/augmented"
        isAugFolderExist = os.path.exists(augmentation_folder)
        if not isAugFolderExist:
            for i in labels:
                os.makedirs(f"{augmentation_folder}/{i}", exist_ok=True)

        # Save augmented images for viewing purpose
        for idx in df_mel_augmented.index:
            # img = Image.fromarray(df_mel_augmented.image[idx], mode='RGB')
            img = array_to_img(df_mel_augmented.image[idx])
            currentPath = pathlib.Path(df_mel_augmented.path[idx])
            label = df_mel_augmented.cell_type_binary[idx]
            assert label == 'Melanoma'
            img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)

        for idx in df_non_mel_augmented.index:
            # img = Image.fromarray(df_non_mel_augmented.image[idx], mode='RGB')
            img = array_to_img(df_non_mel_augmented.image[idx])
            currentPath = pathlib.Path(df_non_mel_augmented.path[idx])
            label = df_non_mel_augmented.cell_type_binary[idx]
            assert label == 'Non-Melanoma'
            img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)
    
        trainpixels_augmented = list(map(lambda x:x, df_trainset_augmented.image)) # Filter out only pixel from the list

        # new_means, new_stds = getMeanStd(trainpixels_HAM10000_augmented)
        if networktype.name == mel.NetworkType.ResNet50.name:
            imgs_augmented = self.normalizeImgs_ResNet50(trainpixels_augmented)
        elif networktype.name == mel.NetworkType.Xception.name:
            imgs_augmented = self.normalizeImgs_Xception(trainpixels_augmented)
        elif networktype.name == mel.NetworkType.InceptionV3.name:
            imgs_augmented = self.normalizeImgs_inceptionV3(trainpixels_augmented)
        elif networktype.name == mel.NetworkType.VGG16.name:
            imgs_augmented = self.normalizeImgs_vgg16(trainpixels_augmented)
        elif networktype.name == mel.NetworkType.VGG19.name:
            imgs_augmented = self.normalizeImgs_vgg19(trainpixels_augmented)
        trainimages_augmented = np.vstack((trainimages, imgs_augmented))
        
        
        
        # labels_augmented = np.asarray(trainset_HAM10000_augmented.cell_type_binary_idx, dtype='float64')
        labels_augmented = to_categorical(df_trainset_augmented.cell_type_binary_idx, num_classes=2)
        trainlabels_augmented = np.vstack((trainlabels, labels_augmented))

        assert len(trainpixels_augmented) == labels_augmented.shape[0]
        assert trainlabels_augmented.shape[0] == trainimages_augmented.shape[0]
    
        # Save features from train/val/test sets divided into malignant/benign (This is only for viewing purpose)
        # for idx, order in enumerate(trainset_HAM10000.index):
        # 	img = Image.fromarray(trainimages_HAM10000[idx][:,:,::-1].astype("uint8"), mode='RGB') # back to RGB
        # 	label = trainset_HAM10000.cell_type_binary[order]
        # 	assert label == df_HAM10000.cell_type_binary[order]
        # 	img.save(f"{train_feature_folder}/{label}/{trainset_HAM10000.image[order][2].stem}.jpg", quality=100, subsampling=0)
        # 	# imsave(f"{train_feature_folder}/{trainset_HAM10000.image[order][2].stem}.tiff",trainimages_HAM10000[idx][:,:,::-1].astype("uint8"))

        # for idx, order in enumerate(validationset_HAM10000.index):
        # 	img = Image.fromarray(validationimages_HAM10000[idx][:,:,::-1].astype("uint8"), mode='RGB')
        # 	label = validationset_HAM10000.cell_type_binary[order]
        # 	assert label == df_HAM10000.cell_type_binary[order]
        # 	img.save(f"{val_feature_folder}/{label}/{validationset_HAM10000.image[order][2].stem}.jpg", quality=100, subsampling=0)

        # for idx, order in enumerate(testset_HAM10000.index):
        # 	img = Image.fromarray(testimages_HAM10000[idx][:,:,::-1].astype("uint8"), mode='RGB')
        # 	label = testset_HAM10000.cell_type_binary[order]
        # 	assert label == df_HAM10000.cell_type_binary[order]
        # 	img.save(f"{test_feature_folder}/{label}/{testset_HAM10000.image[order][2].stem}.jpg", quality=100, subsampling=0)
        
        return datasettype, df_mel_augmented, df_non_mel_augmented, trainimages_augmented, trainlabels_augmented
    

        # Unpack all image pixels using asterisk(*) with dimension (shape[0])
        # trainimages_HAM10000_augmented = trainimages_HAM10000_augmented.reshape(trainimages_HAM10000_augmented.shape[0], *image_shape)

    # def splitDataset(self, datasettype):
    #         ISIC2018_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC-2017_Training_Data')
	# 		ISIC2018_val_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC-2017_Validation_Data')
	# 		ISIC2018_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC-2017_Test_v2_Data')

	# 		num_train_img_ISIC2018 = len(list(ISIC2017_training_path.glob('./*.jpg'))) # counts all ISIC2017 training images
	# 		num_val_img_ISIC2018 = len(list(ISIC2017_val_path.glob('./*.jpg'))) # counts all ISIC2017 validation images
	# 		num_test_img_ISIC2018 = len(list(ISIC2017_test_path.glob('./*.jpg'))) # counts all ISIC2017 test images

	# 		assert num_train_img_ISIC2018 == 10015
	# 		assert num_val_img_ISIC2018 == 193
	# 		assert num_test_img_ISIC2018 == 1512

	# 		logger.debug('%s %s', f"Images available in {datasettype.value} train dataset:", num_train_img_ISIC2018)
	# 		logger.debug('%s %s', f"Images available in {datasettype.value} validation dataset:", num_val_img_ISIC2018)
	# 		logger.debug('%s %s', f"Images available in {datasettype.value} test dataset:", num_test_img_ISIC2018)

	# 		# ISIC2018: Dictionary for Image Names
	# 		imageid_path_training_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_training_path, '*.jpg'))}
	# 		imageid_path_val_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_val_path, '*.jpg'))}
	# 		imageid_path_test_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_test_path, '*.jpg'))}

			
	# 		df_training_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2018', './ISIC-2017_Training_Part3_GroundTruth.csv')))
	# 		df_val_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2018', './ISIC-2017_Validation_Part3_GroundTruth.csv')))
	# 		df_test_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2018', './ISIC-2017_Test_v2_Part3_GroundTruth.csv')))


	# 		logger.debug("Let's check ISIC2017 metadata briefly")
	# 		logger.debug("This is ISIC2017 training data samples")
	# 		# No need to create column titles (1st row) as ISIC2017 has default column titles
	# 		display(df_training_ISIC2017.head())
	# 		logger.debug("This is ISIC2017 test data samples")
	# 		display(df_test_ISIC2017.head())

	# 		classes_ISIC2017_task3_1 = ['nevus or seborrheic keratosis', 'melanoma']
	# 		classes_ISIC2017_task3_2 = ['melanoma or nevus', 'seborrheic keratosis']

	# 		# ISIC2017: Creating New Columns for better readability
	# 		df_training_ISIC2017['path'] = df_training_ISIC2017.image_id.map(imageid_path_training_dict_ISIC2017.get)
	# 		df_training_ISIC2017['cell_type_binary'] = df_training_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
	# 		df_training_ISIC2017['cell_type_task3_1'] = df_training_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
	# 		df_training_ISIC2017['cell_type_task3_2'] = df_training_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
	# 		df_training_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_binary, categories=classes_melanoma_binary).codes
	# 		df_training_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
	# 		df_training_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes

	# 		df_val_ISIC2017['path'] = df_val_ISIC2017.image_id.map(imageid_path_val_dict_ISIC2017.get)
	# 		df_val_ISIC2017['cell_type_binary'] = df_val_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
	# 		df_val_ISIC2017['cell_type_task3_1'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
	# 		df_val_ISIC2017['cell_type_task3_2'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
	# 		df_val_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_binary, categories=classes_melanoma_binary).codes
	# 		df_val_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
	# 		df_val_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes

	# 		df_test_ISIC2017['path'] = df_test_ISIC2017.image_id.map(imageid_path_test_dict_ISIC2017.get)
	# 		df_test_ISIC2017['cell_type_binary'] = df_test_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
	# 		df_test_ISIC2017['cell_type_task3_1'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
	# 		df_test_ISIC2017['cell_type_task3_2'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
	# 		df_test_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_binary, categories=classes_melanoma_binary).codes
	# 		df_test_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
	# 		df_test_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes



	# 		logger.debug("Check null data in ISIC2017 training metadata")
	# 		display(df_training_ISIC2017.isnull().sum())
	# 		logger.debug("Check null data in ISIC2017 validation metadata")
	# 		display(df_val_ISIC2017.isnull().sum())
	# 		logger.debug("Check null data in ISIC2017 test metadata")
	# 		display(df_test_ISIC2017.isnull().sum())


	# 		# df_training_ISIC2017['ori_image'] = df_training_ISIC2017.path.map(
	# 		# 	lambda x:(
	# 		# 		img := Image.open(x), # [0]: PIL object
	# 		# 		np.asarray(img), # [1]: pixel array
	# 		# 	)
	# 		# )
			
	# 		df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(
	# 			lambda x:(
	# 				# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
	# 				img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
	# 				np.asarray(img), # [1]: pixel array
	# 				currentPath := pathlib.Path(x), # [2]: PosixPath
	# 				# img.save(f"{whole_rgb_folder}/{currentPath.name}")
	# 			)
	# 		)

	# 		# df_val_ISIC2017['ori_image'] = df_val_ISIC2017.path.map(
	# 		# 	lambda x:(
	# 		# 		img := Image.open(x), # [0]: PIL object
	# 		# 		np.asarray(img), # [1]: pixel array
	# 		# 	)
	# 		# )
			
	# 		df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(
	# 			lambda x:(
	# 				# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
	# 				img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
	# 				np.asarray(img), # [1]: pixel array
	# 				currentPath := pathlib.Path(x), # [2]: PosixPath
	# 				# img.save(f"{whole_rgb_folder}/{currentPath.name}")
	# 			)
	# 		)

	# 		# df_test_ISIC2017['ori_image'] = df_test_ISIC2017.path.map(
	# 		# 	lambda x:(
	# 		# 		img := Image.open(x), # [0]: PIL object
	# 		# 		np.asarray(img), # [1]: pixel array
	# 		# 	)
	# 		# )
			
	# 		df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(
	# 			lambda x:(
	# 				# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
	# 				img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
	# 				np.asarray(img), # [1]: pixel array
	# 				currentPath := pathlib.Path(x), # [2]: PosixPath
	# 				# img.save(f"{whole_rgb_folder}/{currentPath.name}")
	# 			)
	# 		)

	# 		assert all(df_training_ISIC2017.cell_type_binary.unique() == df_test_ISIC2017.cell_type_binary.unique())
	# 		assert all(df_val_ISIC2017.cell_type_binary.unique() == df_test_ISIC2017.cell_type_binary.unique())
	# 		labels = df_training_ISIC2017.cell_type_binary.unique()

	# 		if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
	# 			for i in labels:
	# 				os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
	# 				os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
	# 				os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
	# 				os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
	# 		if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
	# 			for i in labels:
	# 				os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
	# 				os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
	# 				os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
	# 				os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)


	# 		# df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
	# 		# df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
	# 		# df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

	# 		# ISIC2017 datasets are divided into train/val/test already
	# 		trainset_ISIC2017 = df_training_ISIC2017
	# 		validationset_ISIC2017 = df_val_ISIC2017
	# 		testset_ISIC2017 = df_test_ISIC2017

	# 		preprocessor.saveNumpyImagesToFiles(trainset_ISIC2017, df_training_ISIC2017, train_rgb_folder)
	# 		preprocessor.saveNumpyImagesToFiles(validationset_ISIC2017, df_val_ISIC2017, val_rgb_folder)
	# 		preprocessor.saveNumpyImagesToFiles(testset_ISIC2017, df_test_ISIC2017, test_rgb_folder)

	# 		# ISIC2017 binary images/labels
	# 		trainpixels_ISIC2017 = list(map(lambda x:x[1], trainset_ISIC2017.image)) # Filter out only pixel from the list
	# 		validationpixels_ISIC2017 = list(map(lambda x:x[1], validationset_ISIC2017.image)) # Filter out only pixel from the list
	# 		testpixels_ISIC2017 = list(map(lambda x:x[1], testset_ISIC2017.image)) # Filter out only pixel from the list
			
	# 		if networktype.name == NetworkType.ResNet50.name:
	# 			trainimages_ISIC2017 = preprocessor.normalizeImgs_ResNet50(trainpixels_ISIC2017)
	# 			testimages_ISIC2017 = preprocessor.normalizeImgs_ResNet50(testpixels_ISIC2017)
	# 			validationimages_ISIC2017 = preprocessor.normalizeImgs_ResNet50(validationpixels_ISIC2017)
	# 		elif networktype.name == NetworkType.Xception.name:
	# 			trainimages_ISIC2017 = preprocessor.normalizeImgs_Xception(trainpixels_ISIC2017)
	# 			testimages_ISIC2017 = preprocessor.normalizeImgs_Xception(testpixels_ISIC2017)
	# 			validationimages_ISIC2017 = preprocessor.normalizeImgs_Xception(validationpixels_ISIC2017)
	# 		elif networktype.name == NetworkType.InceptionV3.name:
	# 			trainimages_ISIC2017 = preprocessor.normalizeImgs_inceptionV3(trainpixels_ISIC2017)
	# 			testimages_ISIC2017 = preprocessor.normalizeImgs_inceptionV3(testpixels_ISIC2017)
	# 			validationimages_ISIC2017 = preprocessor.normalizeImgs_inceptionV3(validationpixels_ISIC2017)
	# 		elif networktype.name == NetworkType.VGG16.name:
	# 			trainimages_ISIC2017 = preprocessor.normalizeImgs_vgg16(trainpixels_ISIC2017)
	# 			testimages_ISIC2017 = preprocessor.normalizeImgs_vgg16(testpixels_ISIC2017)
	# 			validationimages_ISIC2017 = preprocessor.normalizeImgs_vgg16(validationpixels_ISIC2017)
	# 		elif networktype.name == NetworkType.VGG19.name:
	# 			trainimages_ISIC2017 = preprocessor.normalizeImgs_vgg19(trainpixels_ISIC2017)
	# 			testimages_ISIC2017 = preprocessor.normalizeImgs_vgg19(testpixels_ISIC2017)
	# 			validationimages_ISIC2017 = preprocessor.normalizeImgs_vgg19(validationpixels_ISIC2017)
	# 		# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
	# 		# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
	# 		# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
	# 		trainlabels_binary_ISIC2017 = to_categorical(trainset_ISIC2017.cell_type_binary_idx, num_classes=2)
	# 		testlabels_binary_ISIC2017 = to_categorical(testset_ISIC2017.cell_type_binary_idx, num_classes=2)
	# 		validationlabels_binary_ISIC2017 = to_categorical(validationset_ISIC2017.cell_type_binary_idx, num_classes=2)

	# 		assert num_train_img_ISIC2017 == len(trainpixels_ISIC2017)
	# 		assert num_val_img_ISIC2017 == len(validationpixels_ISIC2017)
	# 		assert num_test_img_ISIC2017 == len(testpixels_ISIC2017)
	# 		assert len(trainpixels_ISIC2017) == trainlabels_binary_ISIC2017.shape[0]
	# 		assert len(validationpixels_ISIC2017) == validationlabels_binary_ISIC2017.shape[0]
	# 		assert len(testpixels_ISIC2017) == testlabels_binary_ISIC2017.shape[0]
	# 		assert trainimages_ISIC2017.shape[0] == trainlabels_binary_ISIC2017.shape[0]
	# 		assert validationimages_ISIC2017.shape[0] == validationlabels_binary_ISIC2017.shape[0]
	# 		assert testimages_ISIC2017.shape[0] == testlabels_binary_ISIC2017.shape[0]
	# 		# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

	# 		assert datasettype.name == 'ISIC2017'
	# 		filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
	# 		with open(filename, 'wb') as file_bin:
				
	# 			pickle.dump((trainimages_ISIC2017, testimages_ISIC2017, validationimages_ISIC2017,
	# 			trainlabels_binary_ISIC2017, testlabels_binary_ISIC2017, validationlabels_binary_ISIC2017,
	# 			2), file_bin)
	# 		file_bin.close()

	# 		if augment_ratio is not None and augment_ratio >= 1.0:
				
	# 			augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_ISIC2017_augmented, trainlabels_binary_ISIC2017_augmented = \
	# 				preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_ISIC2017, trainlabels_binary_ISIC2017, \
	# 					augment_ratio, df_training_ISIC2017)
				
	# 			assert augmented_db_name.name == 'ISIC2017'
	# 			filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
	# 			with open(filename_bin, 'wb') as file_bin:
					
	# 				pickle.dump((trainimages_ISIC2017_augmented, testimages_ISIC2017, validationimages_ISIC2017,
	# 				trainlabels_binary_ISIC2017_augmented, testlabels_binary_ISIC2017, validationlabels_binary_ISIC2017,
	# 				2), file_bin)
	# 			file_bin.close()

    def saveNumpyImagesToFiles(self, df, original_df, base_path):
        def assert_(cond): assert cond
        return df.index.map(lambda x: (
				currentPath_train := pathlib.Path(df.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
				label := df.cell_type_binary[x],
				assert_(label == original_df.cell_type_binary[x]),
				df.image[x][0].save(f"{base_path}/{label}/{currentPath_train.name}", quality=100, subsampling=0)
			))

