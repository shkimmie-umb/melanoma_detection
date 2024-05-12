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

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


	

class Preprocess:

    def __init__(self):
        # self.square_size = square_size
        # desired size
        # if image_size is not None:
        #     self.img_width = image_size[1]
        #     self.img_height = image_size[0]

        self.preprMethodDict = {
            # RGB to BGR
            mel.NetworkType.ResNet50.name: tf.keras.applications.resnet.preprocess_input,
            mel.NetworkType.ResNet101.name: tf.keras.applications.resnet.preprocess_input,
            mel.NetworkType.ResNet152.name: tf.keras.applications.resnet.preprocess_input,
            mel.NetworkType.Xception.name: tf.keras.applications.xception.preprocess_input,
            mel.NetworkType.InceptionV3.name: tf.keras.applications.inception_v3.preprocess_input,
            mel.NetworkType.VGG16.name: tf.keras.applications.vgg16.preprocess_input,
            mel.NetworkType.VGG19.name: tf.keras.applications.vgg19.preprocess_input,
            mel.NetworkType.EfficientNetB0.name: tf.keras.applications.efficientnet.preprocess_input,
            mel.NetworkType.EfficientNetB1.name: tf.keras.applications.efficientnet.preprocess_input,
            mel.NetworkType.EfficientNetB2.name: tf.keras.applications.efficientnet.preprocess_input,
            mel.NetworkType.EfficientNetB3.name: tf.keras.applications.efficientnet.preprocess_input,
            mel.NetworkType.EfficientNetB4.name: tf.keras.applications.efficientnet.preprocess_input,
            mel.NetworkType.EfficientNetB5.name: tf.keras.applications.efficientnet.preprocess_input,
            mel.NetworkType.EfficientNetB6.name: tf.keras.applications.efficientnet.preprocess_input,
            mel.NetworkType.EfficientNetB7.name: tf.keras.applications.efficientnet.preprocess_input,

            mel.NetworkType.ResNet50V2.name: tf.keras.applications.resnet_v2.preprocess_input,
            mel.NetworkType.ResNet101V2.name: tf.keras.applications.resnet_v2.preprocess_input,
            mel.NetworkType.ResNet152V2.name: tf.keras.applications.resnet_v2.preprocess_input,

            mel.NetworkType.MobileNet.name: tf.keras.applications.mobilenet.preprocess_input,
            mel.NetworkType.MobileNetV2.name: tf.keras.applications.mobilenet_v2.preprocess_input,

            mel.NetworkType.DenseNet121.name: tf.keras.applications.densenet.preprocess_input,
            mel.NetworkType.DenseNet169.name: tf.keras.applications.densenet.preprocess_input,
            mel.NetworkType.DenseNet201.name: tf.keras.applications.densenet.preprocess_input,

            mel.NetworkType.NASNetMobile.name: tf.keras.applications.nasnet.preprocess_input,
            mel.NetworkType.NASNetLarge.name: tf.keras.applications.nasnet.preprocess_input,
        }

    @staticmethod
    def normalizeImg(img):
        img = img/255.0

        return img
        
    
    def normalizeImgs(self, imgs, networktype, uniform_normalization):
        imgList = []
        for img in imgs:
            img = np.expand_dims(img, axis=0)
            if uniform_normalization is False:
                transformed_img = self.preprMethodDict[networktype.name](img)
            elif uniform_normalization is True:
                norm = A.Normalize(mean=0.0, std=1.0)
                transformed_img = norm(image=img)['image']
            # transformed_img = tf.keras.applications.resnet.preprocess_input(img) # RGB to BGR
            # imgList.append(transformed_img/255.)
            imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG

    def Dataset_loader(self, imgPath, networktype, debug_path=None):
        imgList = []
        read = lambda imname: load_img(path=imname, target_size=(self.img_height, self.img_width))
        imgPathList = os.listdir(imgPath)
        for idx, IMAGE_NAME in enumerate(tqdm(imgPathList)):
            PATH = os.path.join(imgPath,IMAGE_NAME)
            _, ftype = os.path.splitext(PATH)
            
            label = os.path.basename(os.path.normpath(imgPath)) # Extract label from folder name
            if ftype.lower() == ".jpg" or ftype.lower() == ".jpeg" or ftype.lower() == ".bmp" or ftype.lower() == ".png":
                img = read(PATH)
                img.save(f"{debug_path}/{label}/{IMAGE_NAME}", quality=100, subsampling=0)
                img = np.expand_dims(img, axis=0)
                transformed_img = self.preprMethodDict[networktype.name](img)
                # transformed_img = tf.keras.applications.resnet.preprocess_input(img)
                # imgList.append(transformed_img/255.)
                imgList.append(transformed_img)
        IMG = np.vstack(imgList)
        return IMG
    

    def squareImgs(self, path, square_size):
        img = load_img(path=path, target_size=None)
        if square_size is not None:
            img_width, img_height = img.size
            left = img.size[0]/2 - min(img_width, img_height)/2
            upper = img.size[1]/2 - min(img_width, img_height)/2
            right = img.size[0]/2 + min(img_width, img_height)/2
            bottom = img.size[1]/2 + min(img_width, img_height)/2

            squared_img = img.crop((left, upper, right, bottom))
            # Make squares and resize them
            squared_img = squared_img.resize(size=(square_size, square_size), resample=Image.LANCZOS)
        elif square_size is None:
            squared_img = img

        return squared_img
    
    def squareImgsAndResize(self, path, square_size, resize_width, resize_height):
        # This makes images square
        squared_img = self.squareImgs(path, square_size)
        
        if resize_width is not None and resize_height is not None:
            
            # This resizes all square images into desired size
            resized_img = squared_img.resize(size=(resize_width, resize_height), resample=Image.LANCZOS)
        else:
            resized_img = squared_img

        return resized_img




    
    def augmentation(self, train_rgb_folder, labels, trainimages, trainlabels, square_size, resize_width, resize_height, augment_ratio, df_trainset):
        
        # augMethod = aug.Augmentation(aug.crop_flip_brightnesscontrast())
        augMethod = aug.Augmentation(aug.crop_flip())


        df_mel = df_trainset[df_trainset.cell_type_binary=='Melanoma']
        df_non_mel = df_trainset[df_trainset.cell_type_binary=='Non-Melanoma']

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
                # img = load_img(path=df_trainset.path[randmel_idx], target_size=None)
                img = self.squareImgs(path=df_trainset.path[randmel_idx], square_size=square_size)
                np_img = img_to_array(img)
                df_mel_augmented.loc[j] = df_trainset.loc[randmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                # display(f'path: {df_trainset.path[randmel_idx]}')
                augmented_img = augMethod.augmentation(
                    input_img=np_img,
                    crop_height=resize_height, 
                    crop_width=resize_width, 
                    zoomout=1.0, 
                    zoomin=1.1, 
                    p_scaling=0.5, 
                    p_rotation=0.5, 
                    p_hflip=0.5, 
                    p_vflip=0.5)
                df_mel_augmented.at[j, 'image'] = None
                df_mel_augmented.at[j, 'image'] = mel.Parser.encode(array_to_img(augmented_img['image']))
                num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - (non_mel_cnt - mel_cnt)
                assert df_mel_augmented.shape[0] <= num_augmented_img
                

            # non-melanoma augmentation here
            for j, id in enumerate(range(non_mel_cnt, math.ceil(non_mel_cnt * augment_ratio))):
                randnonmel_idx = random.choice(df_non_mel.index)
                assert df_non_mel.path[randnonmel_idx] == df_trainset.path[randnonmel_idx]
                # img = Image.open(df_trainset.path[randnonmel_idx]).convert("RGB")
                # img = load_img(path=df_trainset.path[randnonmel_idx], target_size=None)
                img = self.squareImgs(path=df_trainset.path[randmel_idx], square_size=square_size)
                np_img = img_to_array(img)
                df_non_mel_augmented.loc[j] = df_trainset.loc[randnonmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                augmented_img = augMethod.augmentation(
                    input_img=np_img, 
                    crop_height=resize_height, 
                    crop_width=resize_width, 
                    zoomout=1.0, 
                    zoomin=1.1, 
                    p_scaling=0.5, 
                    p_rotation=0.5, 
                    p_hflip=0.5, 
                    p_vflip=0.5)
                df_non_mel_augmented.at[j, 'image'] = None
                df_non_mel_augmented.at[j, 'image'] = mel.Parser.encode(array_to_img(augmented_img['image']))
                num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - non_mel_cnt
                assert df_non_mel_augmented.shape[0] <= num_augmented_img
        elif mel_cnt > non_mel_cnt:
            # melanoma augmentation here
            for j, id in enumerate(range(mel_cnt, math.ceil(mel_cnt * augment_ratio))):
                randmel_idx = random.choice(df_mel.index)
                assert df_mel.path[randmel_idx] == df_trainset.path[randmel_idx]
                # img = Image.open(df_trainset.path[randmel_idx]).convert("RGB")
                # img = load_img(path=df_trainset.path[randmel_idx], target_size=None)
                img = self.squareImgs(path=df_trainset.path[randmel_idx], square_size=square_size)
                np_img = img_to_array(img)
                df_mel_augmented.loc[j] = df_trainset.loc[randmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                augmented_img = augMethod.augmentation(
                    input_img=np_img, 
                    crop_height=resize_height, 
                    crop_width=resize_width, 
                    zoomout=1.0, 
                    zoomin=1.1, 
                    p_scaling=0.5, 
                    p_rotation=0.5, 
                    p_hflip=0.5, 
                    p_vflip=0.5)
                df_mel_augmented.at[j, 'image'] = None
                df_mel_augmented.at[j, 'image'] = mel.Parser.encode(array_to_img(augmented_img['image']))
                num_augmented_img = math.ceil(mel_cnt * augment_ratio) - mel_cnt
                assert df_mel_augmented.shape[0] <= num_augmented_img

            # non-melanoma augmentation here
            for j, id in enumerate(range((mel_cnt - non_mel_cnt), math.ceil(mel_cnt * augment_ratio))):
                randnonmel_idx = random.choice(df_non_mel.index)
                assert df_non_mel.path[randnonmel_idx] == df_trainset.path[randnonmel_idx]
                # img = Image.open(df_trainset.path[randnonmel_idx]).convert("RGB")
                # img = load_img(path=df_trainset.path[randnonmel_idx], target_size=None)
                img = self.squareImgs(path=df_trainset.path[randmel_idx], square_size=square_size)
                # np_img = np.asarray(img)
                np_img = img_to_array(img)
                df_non_mel_augmented.loc[j] = df_trainset.loc[randnonmel_idx]
                # augmented_img = augMethod.augmentation(input_img=np_img, crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
                # 										p_randomBrightnessContrast=0.2)
                augmented_img = augMethod.augmentation(input_img=np_img, 
                                                       crop_height=resize_height, 
                                                       crop_width=resize_width, 
                                                       zoomout=1.0, 
                                                       zoomin=1.1, 
                                                       p_scaling=0.5, 
                                                       p_rotation=0.5, 
                                                       p_hflip=0.5, 
                                                       p_vflip=0.5)
                df_non_mel_augmented.at[j, 'image'] = None
                df_non_mel_augmented.at[j, 'image'] = mel.Parser.encode(array_to_img(augmented_img['image']))
                num_augmented_img = math.ceil(mel_cnt * augment_ratio) - (mel_cnt - non_mel_cnt)
                assert df_non_mel_augmented.shape[0] <= num_augmented_img

        df_trainset_augmented = pd.concat([df_mel_augmented, df_non_mel_augmented], ignore_index=True, axis=0)

        augmentation_folder = f"{train_rgb_folder}/augmented"
        isAugFolderExist = os.path.exists(augmentation_folder)
        if not isAugFolderExist:
            for i in labels:
                os.makedirs(f"{augmentation_folder}/{i}", exist_ok=True)

        # Save augmented images for viewing purpose
        for idx in df_mel_augmented.index:
            # img = Image.fromarray(df_mel_augmented.image[idx], mode='RGB')
            img = mel.Parser.decode(df_mel_augmented['image'][idx])
            currentPath = pathlib.Path(df_mel_augmented.path[idx])
            label = df_mel_augmented.cell_type_binary[idx]
            assert label == 'Melanoma'
            img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg")

        for idx in df_non_mel_augmented.index:
            # img = Image.fromarray(df_non_mel_augmented.image[idx], mode='RGB')
            img = mel.Parser.decode(df_non_mel_augmented['image'][idx])
            currentPath = pathlib.Path(df_non_mel_augmented.path[idx])
            label = df_non_mel_augmented.cell_type_binary[idx]
            assert label == 'Non-Melanoma'
            img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg")
    
        trainpixels_augmented = list(map(lambda x:x, df_trainset_augmented['image'])) # Filter out only pixel from the list

        
        # imgs_augmented = self.normalizeImgs(imgs=trainpixels_augmented, networktype=networktype, uniform_normalization=uniform_normalization)

        # trainimages_augmented = np.vstack((trainimages, trainpixels_augmented))
        # trainimages_augmented = pd.concat([trainimages, trainpixels_augmented], ignore_index=True, axis=0)
        trainimages_augmented = trainimages + trainpixels_augmented
        
        # ids augmented
        trainids_augmented = list(map(lambda x:pathlib.Path(x).stem, df_trainset_augmented.path))
        
        
        # labels_augmented = np.asarray(trainset_HAM10000_augmented.cell_type_binary_idx, dtype='float64')
        labels_augmented = to_categorical(df_trainset_augmented.cell_type_binary_idx, num_classes=2)
        trainlabels_augmented = np.vstack((trainlabels, labels_augmented))

        assert len(trainpixels_augmented) == labels_augmented.shape[0]
        assert trainlabels_augmented.shape[0] == len(trainimages_augmented)
    
        
        return df_mel_augmented, df_non_mel_augmented, trainimages_augmented, trainlabels_augmented, trainids_augmented
    



    def saveNumpyImagesToFiles(self, df, original_df, base_path):
        # def assert_(cond): assert cond
        # return df.index.map(lambda x: (
        #         print(df.image[x][1]),
        #         img := load_img(path=df.image[x][1], target_size=None),
		# 		currentPath_train := pathlib.Path(df.image[x][1]), # [0]: Encoded PIL obj, [1]: PosixPath
		# 		label := df.cell_type_binary[x],
		# 		assert_(label == original_df.cell_type_binary[x]),
		# 		img.save(f"{base_path}/{label}/{currentPath_train.name}", quality=100, subsampling=0)
		# 	))
        for idx, sliced_idx in enumerate(df.index):
            # Print will make the logic fail
            # print(df.image[sliced_idx][1])
            img = mel.Parser.decode(df['image'][sliced_idx][0])
            currentPath = pathlib.Path(df['image'][sliced_idx][1])
            label = df.cell_type_binary[sliced_idx]
            assert label == original_df.cell_type_binary[sliced_idx]
            img.save(f"{base_path}/{label}/{currentPath.stem}.jpg")

    
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