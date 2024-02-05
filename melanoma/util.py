from glob import glob
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from enum import Enum
import seaborn as sns
import pickle

import PIL
from PIL import Image
from tqdm import tqdm
import json


from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img
from .preprocess import Preprocess


import matplotlib.pyplot as plt


from IPython.core.debugger import Pdb
from IPython.display import display

import logging


class NetworkType(Enum):
	ResNet50 = 1
	Xception = 2
	InceptionV3 = 3
	VGG16 = 4
	VGG19 = 5

class DatasetType(Enum):
	HAM10000 = 1
	ISIC2016= 2
	ISIC2017=3
	ISIC2018 = 4
	ISIC2019 = 5
	ISIC2020 = 6
	PH2 = 7
	_7_point_criteria = 8
	PAD_UFES_20 = 9
	MEDNODE = 10
	KaggleMB = 11

class ClassType(Enum):
	multi = 1
	binary = 2

class Util:
	# trainDataPath = ''
	# testDataPath = ''
	# train_ds = ''
	# val_ds = ''
	# class_names = []
	def __init__(self, path, image_size=(None, None)):
		self.base_dir = pathlib.Path(path)
		
		# self.trainDataPath = pathlib.Path.joinpath(path, '/Train')
		# self.testDataPath = pathlib.Path.joinpath(path, '/Test')
		# self.seed_val = seed_val
		# self.split_portion = split_portion
		self.image_size = image_size # (height, width)
		# self.batch_size = batch_size
		# self.color_mode = color_mode
		# self.epochs = epochs
		self.class_names = []
		# self.class_names = class_names
		self.train_ds = ''
		self.val_ds = ''

		self.classes_melanoma_binary = ['Non-Melanoma', 'Melanoma']

		self.foldersExist = ''
		self.RGBfolders = ''

		self.common_binary_label = {
			0.0: 'Non-Melanoma',
			1.0: 'Melanoma',
		}

		# HAM10000
		self.lesion_type_dict_HAM10000 = {
			'bkl'  : 'Pigmented Benign keratosis',
			'nv'   : 'Melanocytic nevi', # nevus
			'df'   : 'Dermatofibroma',
			'mel'  : 'Melanoma',
			'vasc' : 'Vascular lesions',
			'bcc'  : 'Basal cell carcinoma',
			'akiec': 'Actinic keratoses',
		}

		self.lesion_type_binary_dict_HAM10000 = {
			'bkl'  : 'Non-Melanoma',
			'nv'   : 'Non-Melanoma', # nevus
			'df'   : 'Non-Melanoma',
			'mel'  : 'Melanoma',
			'vasc' : 'Non-Melanoma',
			'bcc'  : 'Non-Melanoma',
			'akiec': 'Non-Melanoma',
		}

		# ISIC2016
		self.lesion_type_binary_dict_training_ISIC2016 = {
			'benign' : 'Non-Melanoma',
			'malignant' : 'Melanoma',
		}
		self.lesion_type_binary_dict_test_ISIC2016 = {
			0.0 : 'Non-Melanoma',
			1.0 : 'Melanoma',
		}

		# ISIC2017
		self.lesion_type_dict_ISIC2017_task3_1 = { # Official ISIC2017 task 3 - 1
			0.0: 'nevus or seborrheic keratosis',
			1.0: 'melanoma'
		}
		self.lesion_type_dict_ISIC2017_task3_2 = { # Official ISIC2017 task 3 - 2
			0.0: 'melanoma or nevus',
			1.0: 'seborrheic keratosis',
		}
		self.lesion_type_binary_dict_ISIC2017 = { # Binary melanoma detection
			0.0: 'Non-Melanoma',
			1.0: 'Melanoma',
		}

		# ISIC2020
		self.lesion_type_binary_dict_training_ISIC2020 = {
			'benign' : 'Non-Melanoma',
			'malignant' : 'Melanoma',
		}





	def loadTrainData(self):
		print("path: ", self.trainDataPath)
		print("seed value: ", self.seed_val)
		print("color_mode: ", self.color_mode)
		# self.trainDataPath = pathlib.Path(path)
		num_train_img = len(list(self.trainDataPath.glob('*/*.jpg'))) # counts all images inside 'Train' folder
		print("Images available in train dataset:", num_train_img)

		# Loading the training data
		# using seed=123 while creating dataset using tf.keras.preprocessing.image_dataset_from_directory
		# resizing images to the size img_height*img_width, while writing the dataset
		print("trainDataPath: ", self.trainDataPath)
		self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(self.trainDataPath,
		                                                               seed=self.seed_val,
		                                                               validation_split=self.split_portion,
		                                                               image_size=self.image_size,
		                                                               batch_size=self.batch_size,
		                                                               color_mode=self.color_mode,
		                                                               subset='training')
		train_ds = self.train_ds

		self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(self.trainDataPath,
		                                                               seed=self.seed_val,
		                                                               validation_split=self.split_portion,
		                                                               image_size=self.image_size,
		                                                               batch_size=self.batch_size,
		                                                               color_mode=self.color_mode,
		                                                               subset='validation')
		val_ds = self.val_ds
		AUTOTUNE = tf.data.experimental.AUTOTUNE
		train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
		val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
		# List out all the classes of training data in a list.
		self.class_names = self.train_ds.class_names
		print("Training classes are: ")
		class_names = self.class_names
		print(class_names)

		return train_ds, val_ds

	
	def generateAvgImages(self, mode):
		if mode.value == DatasetType['HAM10000'].value:
			# Dermatology MNIST: Loading and Processing
			# Python · Skin Cancer MNIST: HAM10000

			self.base_skin_dir = str(pathlib.Path.joinpath(pathlib.Path.cwd(), './HAM10000_images_combined'))
			# self.base_skin_dir = self.base_dir.relative_to('./HAM10000_images_combined')
			imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
						for x in glob(os.path.join(self.base_skin_dir, '*.jpg'))} # Don't need extra *
			tile_df = pd.read_csv(os.path.join(str(self.base_dir) + '/HAM10000_metadata.csv'))
			tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
			tile_df['cell_type'] = tile_df['dx'].map(self.lesion_type_dict_HAM10000.get)
			tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
			tile_df.sample(3)
			tile_df.describe(exclude=[np.number])

			fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
			tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)

			# load in all of the images
			from skimage.io import imread
			tile_df['image'] = tile_df['path'].map(imread)

			# see the image size distribution
			print('see the image size distribution')
			print(tile_df['image'].map(lambda x: x.shape).value_counts())

			# Show off a few in each category
			n_samples = 5
			fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
			for n_axs, (type_name, type_rows) in zip(m_axs, 
													tile_df.sort_values(['cell_type']).groupby('cell_type')):
				n_axs[0].set_title(type_name)
				for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
					c_ax.imshow(c_row['image'])
					c_ax.axis('off')
			fig.savefig('category_samples.png', dpi=300)

			# Get Average Color Information
			# Here we get and normalize all of the color channel information

			rgb_info_df = tile_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v in 
                                  zip(['Red', 'Green', 'Blue'], 
                                      np.mean(x['image'], (0, 1)))}),1)
			gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1)
			for c_col in rgb_info_df.columns:
				rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec
			rgb_info_df['Gray_mean'] = gray_col_vec
			display(rgb_info_df.sample(3))

			for c_col in rgb_info_df.columns:
				tile_df[c_col] = rgb_info_df[c_col].values # we cant afford a copy
			sns.pairplot(tile_df[['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean', 'cell_type']], 
             hue='cell_type', plot_kws = {'alpha': 0.5})


			# Show Color Range
			# Show how the mean color channel values affect images
			for sample_col in ['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean']:
				fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
				def take_n_space(in_rows, val_col, n):
					s_rows = in_rows.sort_values([val_col])
					s_idx = np.linspace(0, s_rows.shape[0]-1, n, dtype=int)
					return s_rows.iloc[s_idx]
				for n_axs, (type_name, type_rows) in zip(m_axs, 
														tile_df.sort_values(['cell_type']).groupby('cell_type')):

					for c_ax, (_, c_row) in zip(n_axs, 
												take_n_space(type_rows, 
															sample_col,
															n_samples).iterrows()):
						c_ax.imshow(c_row['image'])
						c_ax.axis('off')
						c_ax.set_title('{:2.2f}'.format(c_row[sample_col]))
					n_axs[0].set_title(type_name)
				fig.savefig('{}_samples.png'.format(sample_col), dpi=300)

			# Make a nice cover image
			# Make a cover image for the dataset using all of the tiles

			from skimage.util import montage
			rgb_stack = np.stack(tile_df.\
								sort_values(['cell_type', 'Red_mean'])['image'].\
								map(lambda x: x[::5, ::5]).values, 0)
			rgb_montage = np.stack([montage(rgb_stack[:, :, :, i]) for i in range(rgb_stack.shape[3])], -1)
			print(rgb_montage.shape)

			fig, ax1 = plt.subplots(1, 1, figsize = (20, 20), dpi=300)
			ax1.imshow(rgb_montage)
			fig.savefig('nice_montage.png')

			from skimage.io import imsave
			# this is a big file, imsave('full_dataset_montage.png', rgb_montage)

			# Make an MNIST Like Dataset
			# We can make an MNIST-like dataset by flattening the images into vectors and exporting them
			display(tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates())

			from PIL import Image
			def package_mnist_df(in_rows, 
								image_col_name = 'image',
								label_col_name = 'cell_type_idx',
								image_shape=(28, 28), 
								image_mode='RGB',
								label_first=False
								):
				out_vec_list = in_rows[image_col_name].map(lambda x: 
														np.array(Image.\
																	fromarray(x).\
																	resize(image_shape, resample=Image.LANCZOS).\
																	convert(image_mode)).ravel())
				out_vec = np.stack(out_vec_list, 0)
				out_df = pd.DataFrame(out_vec)
				n_col_names =  ['pixel{:04d}'.format(i) for i in range(out_vec.shape[1])]
				out_df.columns = n_col_names
				out_df['label'] = in_rows[label_col_name].values.copy()
				if label_first:
					return out_df[['label']+n_col_names]
				else:
					return out_df
				
			from itertools import product
			for img_side_dim, img_mode in product([8, 28, 64, 128], ['L', 'RGB']):
				if (img_side_dim==128) and (img_mode=='RGB'):
					# 128x128xRGB is a biggie
					break
				out_df = package_mnist_df(tile_df, 
										image_shape=(img_side_dim, img_side_dim),
										image_mode=img_mode)
				out_path = f'hmnist_{img_side_dim}_{img_side_dim}_{img_mode}.csv'
				out_df.to_csv(out_path, index=False)
				print(f'Saved {out_df.shape} -> {out_path}: {os.stat(out_path).st_size/1024:2.1f}kb')
	
	def saveDatasetsToFile(self, datasettype, networktype, augment_ratio):
		# create logger
		logger = logging.getLogger('Melanoma classification')
		logger.setLevel(logging.DEBUG)

		
		def ExtractPixel(image):
					return [item[1] for item in image]
		
		networkname = networktype.name

		path = str(self.base_dir) + '/melanomaDB' + '/customDB' + '/' + networkname
		# data_gen_HAM10000, HAM10000_multiclass, HAM10000_binaryclass, data_gen_ISIC2016, ISIC2016_binaryclass = self.load(mode)
		isExist = os.path.exists(path)
		if not isExist :
			os.makedirs(path)
		else:
			pass

		print("path: ", self.base_dir)
		
		# Dataset path define
		from datetime import datetime
		now = datetime.now() # current date and time

		date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
		datasetname = datasettype.name
		

		debug_rgb_folder = path + f'/debug/{datasetname}/RGB_resized/'+f'{self.image_size[0]}h_{self.image_size[1]}w_{date_time}'
		debug_feature_folder = path + f'/debug/{datasetname}/feature/'+f'{self.image_size[0]}h_{self.image_size[1]}w_{date_time}'
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
		whole_rgb_folder = debug_rgb_folder + '/Whole'
		train_rgb_folder = debug_rgb_folder + '/Train'
		val_rgb_folder = debug_rgb_folder + '/Val'
		test_rgb_folder = debug_rgb_folder + '/Test'

		whole_feature_folder = debug_feature_folder + '/Whole'
		train_feature_folder = debug_feature_folder + '/Train'
		val_feature_folder = debug_feature_folder + '/Val'
		test_feature_folder = debug_feature_folder + '/Test'

		

		isWholeRGBExist = os.path.exists(whole_rgb_folder)
		isTrainRGBExist = os.path.exists(train_rgb_folder)
		isValRGBExist = os.path.exists(val_rgb_folder)
		isTestRGBExist = os.path.exists(test_rgb_folder)
		isWholeFeatureExist = os.path.exists(whole_feature_folder)
		isTrainFeatureExist = os.path.exists(train_feature_folder)
		isValFeatureExist = os.path.exists(val_feature_folder)
		isTestFeatureExist = os.path.exists(test_feature_folder)


		

		# df = pd.read_pickle(f"../input/skin-cancer-mnist-ham10000-pickle/HAM10000_metadata-h{CFG['img_height']}-w{CFG['img_width']}.pkl")
		pd.set_option('display.max_columns', 500)

		# Given lesion types
		

		# Not required for pickled data
		# resize() order: (width, height)
		img_height = self.image_size[0]
		img_width = self.image_size[1]

		preprocessor = Preprocess(self.image_size)

		def getMeanStd(trainingimages):
			# images is a list of images
			trainingimages = np.asarray(trainingimages).astype(np.float64)
			trainingimages = trainingimages[:, :, :, ::-1] # RGB to BGR

			meanB = np.mean(trainingimages[:, :, :, 0])
			meanG = np.mean(trainingimages[:, :, :, 1])
			meanR = np.mean(trainingimages[:, :, :, 2])
			stdB = np.std(trainingimages[:, :, :, 0])
			stdG = np.std(trainingimages[:, :, :, 1])
			stdR = np.std(trainingimages[:, :, :, 2])

			means = (meanB, meanG, meanR)
			stds = (stdB, stdG, stdR)
			return means, stds
		# df.to_pickle(f"HAM10000_metadata-h{CFG['img_height']}-w{CFG['img_width']}.pkl", compression='infer', protocol=4)


		# # height, width 순서
		# image_shape = (img_height, img_width, 3)
		#Transfer 'jpg' images to an array IMG
		# def normalizeImgs(imgs):
		# 	img_width = self.image_size[1]
		# 	img_height = self.image_size[0]

		# 	imgList = []
		# 	for img in imgs:
		# 		transformed_img = np.expand_dims(img, axis=0)
		# 		transformed_img = preprocess_input_resnet50(transformed_img)
		# 		transformed_img = transformed_img/255.
		# 		imgList.append(transformed_img)
		# 		# if idx == 0:
		# 		# 	IMG = transformed_img
		# 		# elif idx > 0:
		# 		# 	IMG = np.vstack((IMG, transformed_img))
		# 	IMG = np.vstack(imgList)
		# 	return IMG
		# def prep_cnn_data(num, n_x, n_c, path):
		


		# HAM10000 multi-class images/labels
		if datasettype.value == DatasetType.HAM10000.value:
			HAM10000_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './HAM10000_images_combined')
			num_train_img_HAM10000 = len(list(HAM10000_path.glob('./*.jpg'))) # counts all HAM10000 images

			logger.debug('%s %s', "Images available in HAM10000 train dataset:", num_train_img_HAM10000)

			# HAM10000: Dictionary for Image Names
			imageid_path_dict_HAM10000 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(HAM10000_path, '*.jpg'))}

			df_HAM10000 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './HAM10000_metadata.csv')))

			logger.debug("Let's check HAM10000 metadata briefly -> df.head()")
			# logger.debug("Let's check metadata briefly -> df.head()".format(df.head()))
			# print("Let's check metadata briefly -> df.head()")
			display(df_HAM10000.head())

			classes_multi_HAM10000 = df_HAM10000.dx.unique() # dx column has labels
			num_classes_multi_HAM10000 = len(classes_multi_HAM10000)
			# self.CFG_num_classes = num_classes
			classes_multi_HAM10000, num_classes_multi_HAM10000

			# Not required for pickled data
			# HAM10000: Creating New Columns for better readability
			df_HAM10000['num_images'] = df_HAM10000.groupby('lesion_id')["image_id"].transform("count")
			df_HAM10000['path'] = df_HAM10000.image_id.map(imageid_path_dict_HAM10000.get)
			df_HAM10000['cell_type'] = df_HAM10000.dx.map(self.lesion_type_dict_HAM10000.get)
			df_HAM10000['cell_type_binary'] = df_HAM10000.dx.map(self.lesion_type_binary_dict_HAM10000.get)

			# Define codes for compatibility among datasets
			# df_HAM10000['cell_type_idx'] = pd.Categorical(df_HAM10000.dx).codes
			df_HAM10000['cell_type_idx'] = pd.CategoricalIndex(df_HAM10000.dx, categories=['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']).codes
			# df_HAM10000['cell_type_binary_idx'] = pd.Categorical(df_HAM10000.cell_type_binary).codes
			df_HAM10000['cell_type_binary_idx'] = pd.CategoricalIndex(df_HAM10000.cell_type_binary, categories=self.classes_melanoma_binary).codes
			logger.debug("Let's add some more columns on top of the original metadata for better readability")
			logger.debug("Added columns: 'num_images', 'lesion_id', 'image_id', 'path', 'cell_type', 'cell_type_binary', 'cell_type_idx', 'cell_type_binary_idx'")
			logger.debug("Now, let's show some of records -> df.sample(5)")
			display(df_HAM10000.sample(10))

			# Check null data in metadata
			logger.debug("Check null data in HAM10000 metadata -> df_HAM10000.isnull().sum()")
			display(df_HAM10000.isnull().sum())



			# We found there are some null data in age category
			# Filling in with average data
			logger.debug("HAM10000: We found there are some null data in age category. Let's fill them with average data\n")
			logger.debug("df.age.fillna((df_HAM10000.age.mean()), inplace=True) --------------------")
			df_HAM10000.age.fillna((df_HAM10000.age.mean()), inplace=True)


			# Now, we do not have null data
			logger.debug("HAM10000: Let's check null data now -> print(df.isnull().sum())\n")
			logger.debug("HAM10000: There are no null data as below:")
			display(df_HAM10000.isnull().sum())
			
			df_HAM10000['image'] = df_HAM10000.path.map(
				lambda x:(
				# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
				img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
				# np.asarray(img), # [1]: pixel array
				img_to_array(img), # [1]: pixel array
				currentPath := pathlib.Path(x), # [2]: PosixPath
				# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_HAM10000['img_sizes'] = df_HAM10000.path.map(
				lambda x:(
					Image.open(x).size
				)
			)

			labels = df_HAM10000.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)
				
			

			# Dividing HAM10000 into train/val/test set
			df_single_HAM10000 = df_HAM10000[df_HAM10000.num_images == 1]
			trainset1_HAM10000, testset_HAM10000 = train_test_split(df_single_HAM10000, test_size=0.2,random_state = 2)
			trainset2_HAM10000, validationset_HAM10000 = train_test_split(trainset1_HAM10000, test_size=0.2,random_state = 4)
			trainset3_HAM10000 = df_HAM10000[df_HAM10000.num_images != 1]
			trainset_HAM10000 = pd.concat([trainset2_HAM10000, trainset3_HAM10000])

			preprocessor.saveNumpyImagesToFiles(trainset_HAM10000, df_HAM10000, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(validationset_HAM10000, df_HAM10000, val_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(testset_HAM10000, df_HAM10000, test_rgb_folder)

			# trainset_HAM10000.index.map(lambda x: (
			# 	currentPath_train := pathlib.Path(trainset_HAM10000.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
			# 	label := trainset_HAM10000.cell_type_binary[x],
			# 	assert_(label == df_HAM10000.cell_type_binary[x]),
			# 	trainset_HAM10000.image[x][0].save(f"{train_rgb_folder}/{label}/{currentPath_train.name}", quality=100, subsampling=0)
			# ))

			# validationset_HAM10000.index.map(lambda x: (
			# 	currentPath_val := pathlib.Path(validationset_HAM10000.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
			# 	label := validationset_HAM10000.cell_type_binary[x],
			# 	assert_(label == df_HAM10000.cell_type_binary[x]),
			# 	validationset_HAM10000.image[x][0].save(f"{val_rgb_folder}/{label}/{currentPath_val.name}", quality=100, subsampling=0)
			# ))

			# testset_HAM10000.index.map(lambda x: (
			# 	currentPath_test := pathlib.Path(testset_HAM10000.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
			# 	label := testset_HAM10000.cell_type_binary[x],
			# 	assert_(label == df_HAM10000.cell_type_binary[x]),
			# 	testset_HAM10000.image[x][0].save(f"{test_rgb_folder}/{label}/{currentPath_test.name}", quality=100, subsampling=0)
			# ))

			trainpixels_HAM10000 = list(map(lambda x:x[1], trainset_HAM10000.image)) # Filter out only pixel from the list
			testpixels_HAM10000 = list(map(lambda x:x[1], testset_HAM10000.image))
			validationpixels_HAM10000 = list(map(lambda x:x[1], validationset_HAM10000.image))

			# means, stds = getMeanStd(trainpixels_HAM10000)
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_HAM10000 = preprocessor.normalizeImgs_ResNet50(trainpixels_HAM10000)
				testimages_HAM10000 = preprocessor.normalizeImgs_ResNet50(testpixels_HAM10000)
				validationimages_HAM10000 = preprocessor.normalizeImgs_ResNet50(validationpixels_HAM10000)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_HAM10000 = preprocessor.normalizeImgs_Xception(trainpixels_HAM10000)
				testimages_HAM10000 = preprocessor.normalizeImgs_Xception(testpixels_HAM10000)
				validationimages_HAM10000 = preprocessor.normalizeImgs_Xception(validationpixels_HAM10000)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_HAM10000 = preprocessor.normalizeImgs_inceptionV3(trainpixels_HAM10000)
				testimages_HAM10000 = preprocessor.normalizeImgs_inceptionV3(testpixels_HAM10000)
				validationimages_HAM10000 = preprocessor.normalizeImgs_inceptionV3(validationpixels_HAM10000)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_HAM10000 = preprocessor.normalizeImgs_vgg16(trainpixels_HAM10000)
				testimages_HAM10000 = preprocessor.normalizeImgs_vgg16(testpixels_HAM10000)
				validationimages_HAM10000 = preprocessor.normalizeImgs_vgg16(validationpixels_HAM10000)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_HAM10000 = preprocessor.normalizeImgs_vgg19(trainpixels_HAM10000)
				testimages_HAM10000 = preprocessor.normalizeImgs_vgg19(testpixels_HAM10000)
				validationimages_HAM10000 = preprocessor.normalizeImgs_vgg19(validationpixels_HAM10000)
	
			
			trainlabels_multi_HAM10000 = np.asarray(trainset_HAM10000.cell_type_idx, dtype='float64')
			testlabels_multi_HAM10000 = np.asarray(testset_HAM10000.cell_type_idx, dtype='float64')
			validationlabels_multi_HAM10000 = np.asarray(validationset_HAM10000.cell_type_idx, dtype='float64')
			# trainlabels_binary_HAM10000 = np.asarray(trainset_HAM10000.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_HAM10000 = np.asarray(testset_HAM10000.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_HAM10000 = np.asarray(validationset_HAM10000.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_HAM10000 = to_categorical(trainset_HAM10000.cell_type_binary_idx, num_classes= 2)
			testlabels_binary_HAM10000 = to_categorical(testset_HAM10000.cell_type_binary_idx, num_classes= 2)
			validationlabels_binary_HAM10000 = to_categorical(validationset_HAM10000.cell_type_binary_idx, num_classes= 2)

			assert num_train_img_HAM10000 == (len(trainpixels_HAM10000) + len(testpixels_HAM10000) + len(validationpixels_HAM10000))
			assert len(trainpixels_HAM10000) == trainlabels_multi_HAM10000.shape[0]
			assert len(trainpixels_HAM10000) == trainlabels_binary_HAM10000.shape[0]
			assert len(validationpixels_HAM10000) == validationlabels_multi_HAM10000.shape[0]
			assert len(validationpixels_HAM10000) == validationlabels_binary_HAM10000.shape[0]
			assert len(testpixels_HAM10000) == testlabels_multi_HAM10000.shape[0]
			assert len(testpixels_HAM10000) == testlabels_binary_HAM10000.shape[0]
			assert trainimages_HAM10000.shape[0] == trainlabels_binary_HAM10000.shape[0]
			assert validationimages_HAM10000.shape[0] == validationlabels_binary_HAM10000.shape[0]
			assert testimages_HAM10000.shape[0] == testlabels_binary_HAM10000.shape[0]

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
				

			

			# Unpack all image pixels using asterisk(*) with dimension (shape[0])
			# trainimages_HAM10000 = trainimages_HAM10000.reshape(trainimages_HAM10000.shape[0], *image_shape)
			assert datasettype.name == 'HAM10000'
			filename_bin = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			filename_multi = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_multiclass.pkl' # height x width
			with open(filename_bin, 'wb') as file_bin:
				
				pickle.dump((trainimages_HAM10000, testimages_HAM10000, validationimages_HAM10000,
				trainlabels_binary_HAM10000, testlabels_binary_HAM10000, validationlabels_binary_HAM10000, 2), file_bin)
			file_bin.close()

			# with open(filename_multi, 'wb') as file_multi:
				
			# 	pickle.dump((trainpixels_HAM10000, testpixels_HAM10000, validationpixels_HAM10000,
			# 	trainlabels_multi_HAM10000, testlabels_multi_HAM10000, validationlabels_multi_HAM10000,
			# 	means, stds, num_classes_multi_HAM10000), file_multi)
			# file_multi.close()

			# Augmentation only on training set
			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_HAM10000_augmented, trainlabels_binary_HAM10000_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_HAM10000, trainlabels_binary_HAM10000, \
						augment_ratio, trainset_HAM10000)

				assert augmented_db_name.name == 'HAM10000'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{img_height}h_{img_width}w_binary.pkl' # height x width
        
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_HAM10000_augmented, testimages_HAM10000, validationimages_HAM10000,
					trainlabels_binary_HAM10000_augmented, testlabels_binary_HAM10000, validationlabels_binary_HAM10000,
					2), file_bin)
				file_bin.close()


		
		if datasettype.value == DatasetType.ISIC2016.value:
			ISIC2016_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Training_Data')
			ISIC2016_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Test_Data')

			num_train_img_ISIC2016 = len(list(ISIC2016_training_path.glob('./*.jpg'))) # counts all ISIC2016 training images
			num_test_img_ISIC2016 = len(list(ISIC2016_test_path.glob('./*.jpg'))) # counts all ISIC2016 test images

			assert num_train_img_ISIC2016 == 900
			assert num_test_img_ISIC2016 == 379

			logger.debug('%s %s', "Images available in ISIC2016 train dataset:", num_train_img_ISIC2016)
			logger.debug('%s %s', "Images available in ISIC2016 test dataset:", num_test_img_ISIC2016)

			# ISIC2016: Dictionary for Image Names
			imageid_path_training_dict_ISIC2016 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2016_training_path, '*.jpg'))}
			imageid_path_test_dict_ISIC2016 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2016_test_path, '*.jpg'))}
			ISIC2016_columns = ['image_id', 'label']
			df_training_ISIC2016 = pd.read_csv(str(pathlib.Path.joinpath(
				self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Training_GroundTruth.csv')),
				names=ISIC2016_columns, header=None)
			df_test_ISIC2016 = pd.read_csv(str(pathlib.Path.joinpath(
				self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Test_GroundTruth.csv')),
				names=ISIC2016_columns, header=None)

			logger.debug("Let's check ISIC2016 metadata briefly")
			logger.debug("This is ISIC2016 training data")
			display(df_training_ISIC2016.head())
			logger.debug("This is ISIC2016 test data")
			display(df_test_ISIC2016.head())


			classes_binary_ISIC2016 = df_training_ISIC2016.label.unique() # second column is label
			num_classes_binary_ISIC2016 = len(classes_binary_ISIC2016)
			classes_binary_ISIC2016, num_classes_binary_ISIC2016

			# ISIC2016: Creating New Columns for better readability
			df_training_ISIC2016['path'] = df_training_ISIC2016.image_id.map(imageid_path_training_dict_ISIC2016.get)
			df_training_ISIC2016['cell_type_binary'] = df_training_ISIC2016.label.map(self.lesion_type_binary_dict_training_ISIC2016.get)
			# Define codes for compatibility among datasets
			df_training_ISIC2016['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2016.cell_type_binary, categories=self.classes_melanoma_binary).codes
			df_test_ISIC2016['path'] = df_test_ISIC2016.image_id.map(imageid_path_test_dict_ISIC2016.get)
			df_test_ISIC2016['cell_type_binary'] = df_test_ISIC2016.label.map(self.lesion_type_binary_dict_test_ISIC2016.get)
			# Define codes for compatibility among datasets
			df_test_ISIC2016['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2016.cell_type_binary, categories=self.classes_melanoma_binary).codes
			# logger.debug("Let's add some more columns on top of the original metadata for better readability")
			# logger.debug("Added columns: 'num_images', 'lesion_id', 'image_id', 'path', 'cell_type'")
			# logger.debug("Now, let's show some of records -> df.sample(5)")
			logger.debug("ISIC2016 training df")
			display(df_training_ISIC2016.sample(10))
			logger.debug("ISIC2016 test df")
			display(df_test_ISIC2016.sample(10))

			logger.debug("Check null data in ISIC2016 training metadata -> df_training_ISIC2016.isnull().sum()")
			display(df_training_ISIC2016.isnull().sum())
			logger.debug("Check null data in ISIC2016 test metadata -> df_test_ISIC2016.isnull().sum()")
			display(df_test_ISIC2016.isnull().sum())

			# df_training_ISIC2016['ori_image'] = df_training_ISIC2016.path.map(
			# 	lambda x:(
			# 		img := Image.open(x), # [0]: PIL object
			# 		np.asarray(img), # [1]: pixel array
			# 	)
			# )
			
			df_training_ISIC2016['image'] = df_training_ISIC2016.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_training_ISIC2016['img_sizes'] = df_training_ISIC2016.path.map(
				lambda x:(
					Image.open(x).size
				)
			)

			# df_test_ISIC2016['ori_image'] = df_test_ISIC2016.path.map(
			# 	lambda x:(
			# 		img := Image.open(x), # [0]: PIL object
			# 		np.asarray(img), # [1]: pixel array
			# 	)
			# )
			
			df_test_ISIC2016['image'] = df_test_ISIC2016.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			

			assert all(df_training_ISIC2016.cell_type_binary.unique() == df_test_ISIC2016.cell_type_binary.unique())
			labels = df_training_ISIC2016.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)

			# df_training_ISIC2016['image'] = df_training_ISIC2016.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_test_ISIC2016['image'] = df_test_ISIC2016.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))


			# Dividing ISIC2016 into train/val set
			trainset_ISIC2016, validationset_ISIC2016 = train_test_split(df_training_ISIC2016, test_size=0.2,random_state = 1)
			# ISIC2016 test data is given, so there is no need to create test dataset separately
			testset_ISIC2016 = df_test_ISIC2016


			preprocessor.saveNumpyImagesToFiles(trainset_ISIC2016, df_training_ISIC2016, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(validationset_ISIC2016, df_training_ISIC2016, val_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(testset_ISIC2016, df_test_ISIC2016, test_rgb_folder)


			# ISIC2016 binary images/labels
			trainpixels_ISIC2016 = list(map(lambda x:x[1], trainset_ISIC2016.image)) # Filter out only pixel from the list
			testpixels_ISIC2016 = list(map(lambda x:x[1], testset_ISIC2016.image))
			validationpixels_ISIC2016 = list(map(lambda x:x[1], validationset_ISIC2016.image))

			# means, stds = getMeanStd(trainpixels_ISIC2016)
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_ISIC2016 = preprocessor.normalizeImgs_ResNet50(trainpixels_ISIC2016)
				testimages_ISIC2016 = preprocessor.normalizeImgs_ResNet50(testpixels_ISIC2016)
				validationimages_ISIC2016 = preprocessor.normalizeImgs_ResNet50(validationpixels_ISIC2016)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_ISIC2016 = preprocessor.normalizeImgs_Xception(trainpixels_ISIC2016)
				testimages_ISIC2016 = preprocessor.normalizeImgs_Xception(testpixels_ISIC2016)
				validationimages_ISIC2016 = preprocessor.normalizeImgs_Xception(validationpixels_ISIC2016)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_ISIC2016 = preprocessor.normalizeImgs_inceptionV3(trainpixels_ISIC2016)
				testimages_ISIC2016 = preprocessor.normalizeImgs_inceptionV3(testpixels_ISIC2016)
				validationimages_ISIC2016 = preprocessor.normalizeImgs_inceptionV3(validationpixels_ISIC2016)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_ISIC2016 = preprocessor.normalizeImgs_vgg16(trainpixels_ISIC2016)
				testimages_ISIC2016 = preprocessor.normalizeImgs_vgg16(testpixels_ISIC2016)
				validationimages_ISIC2016 = preprocessor.normalizeImgs_vgg16(validationpixels_ISIC2016)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_ISIC2016 = preprocessor.normalizeImgs_vgg19(trainpixels_ISIC2016)
				testimages_ISIC2016 = preprocessor.normalizeImgs_vgg19(testpixels_ISIC2016)
				validationimages_ISIC2016 = preprocessor.normalizeImgs_vgg19(validationpixels_ISIC2016)
			# trainlabels_binary_ISIC2016 = np.asarray(trainset_ISIC2016.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2016 = np.asarray(testset_ISIC2016.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2016 = np.asarray(validationset_ISIC2016.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_ISIC2016 = to_categorical(trainset_ISIC2016.cell_type_binary_idx, num_classes=2)
			testlabels_binary_ISIC2016 = to_categorical(testset_ISIC2016.cell_type_binary_idx, num_classes=2)
			validationlabels_binary_ISIC2016 = to_categorical(validationset_ISIC2016.cell_type_binary_idx, num_classes=2)

			assert num_train_img_ISIC2016 == (len(trainpixels_ISIC2016) + len(validationpixels_ISIC2016))
			assert num_test_img_ISIC2016 == len(testpixels_ISIC2016)
			assert len(trainpixels_ISIC2016) == trainlabels_binary_ISIC2016.shape[0]
			assert len(validationpixels_ISIC2016) == validationlabels_binary_ISIC2016.shape[0]
			assert len(testpixels_ISIC2016) == testlabels_binary_ISIC2016.shape[0]
			assert trainimages_ISIC2016.shape[0] == trainlabels_binary_ISIC2016.shape[0]
			assert validationimages_ISIC2016.shape[0] == validationlabels_binary_ISIC2016.shape[0]
			assert testimages_ISIC2016.shape[0] == testlabels_binary_ISIC2016.shape[0]

			# trainimages_ISIC2016 = trainimages_ISIC2016.reshape(trainimages_ISIC2016.shape[0], *image_shape)

			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_ISIC2016, testimages_ISIC2016, validationimages_ISIC2016,
				trainlabels_binary_ISIC2016, testlabels_binary_ISIC2016,validationlabels_binary_ISIC2016,
				2), file_bin)
			file_bin.close()


			# Augmentation only on training set
			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_ISIC2016_augmented, trainlabels_binary_ISIC2016_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_ISIC2016, trainlabels_binary_ISIC2016, \
						augment_ratio, df_training_ISIC2016)
				
				assert augmented_db_name.name == 'ISIC2016'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_ISIC2016_augmented, testimages_ISIC2016, validationimages_ISIC2016,
					trainlabels_binary_ISIC2016_augmented, testlabels_binary_ISIC2016, validationlabels_binary_ISIC2016,
					2), file_bin)
				file_bin.close()


		if datasettype.value == DatasetType.ISIC2017.value:
			ISIC2017_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Training_Data')
			ISIC2017_val_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Validation_Data')
			ISIC2017_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Test_v2_Data')

			num_train_img_ISIC2017 = len(list(ISIC2017_training_path.glob('./*.jpg'))) # counts all ISIC2017 training images
			num_val_img_ISIC2017 = len(list(ISIC2017_val_path.glob('./*.jpg'))) # counts all ISIC2017 validation images
			num_test_img_ISIC2017 = len(list(ISIC2017_test_path.glob('./*.jpg'))) # counts all ISIC2017 test images

			assert num_train_img_ISIC2017 == 2000
			assert num_val_img_ISIC2017 == 150
			assert num_test_img_ISIC2017 == 600

			logger.debug('%s %s', "Images available in ISIC2017 train dataset:", num_train_img_ISIC2017)
			logger.debug('%s %s', "Images available in ISIC2017 validation dataset:", num_val_img_ISIC2017)
			logger.debug('%s %s', "Images available in ISIC2017 test dataset:", num_test_img_ISIC2017)

			# ISIC2017: Dictionary for Image Names
			imageid_path_training_dict_ISIC2017 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2017_training_path, '*.jpg'))}
			imageid_path_val_dict_ISIC2017 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2017_val_path, '*.jpg'))}
			imageid_path_test_dict_ISIC2017 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2017_test_path, '*.jpg'))}

			
			df_training_ISIC2017 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Training_Part3_GroundTruth.csv')))
			df_val_ISIC2017 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Validation_Part3_GroundTruth.csv')))
			df_test_ISIC2017 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Test_v2_Part3_GroundTruth.csv')))


			logger.debug("Let's check ISIC2017 metadata briefly")
			logger.debug("This is ISIC2017 training data samples")
			# No need to create column titles (1st row) as ISIC2017 has default column titles
			display(df_training_ISIC2017.head())
			logger.debug("This is ISIC2017 test data samples")
			display(df_test_ISIC2017.head())

			classes_ISIC2017_task3_1 = ['nevus or seborrheic keratosis', 'melanoma']
			classes_ISIC2017_task3_2 = ['melanoma or nevus', 'seborrheic keratosis']

			# ISIC2017: Creating New Columns for better readability
			df_training_ISIC2017['path'] = df_training_ISIC2017.image_id.map(imageid_path_training_dict_ISIC2017.get)
			df_training_ISIC2017['cell_type_binary'] = df_training_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
			df_training_ISIC2017['cell_type_task3_1'] = df_training_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
			df_training_ISIC2017['cell_type_task3_2'] = df_training_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
			df_training_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_binary, categories=self.classes_melanoma_binary).codes
			df_training_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
			df_training_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes

			df_val_ISIC2017['path'] = df_val_ISIC2017.image_id.map(imageid_path_val_dict_ISIC2017.get)
			df_val_ISIC2017['cell_type_binary'] = df_val_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
			df_val_ISIC2017['cell_type_task3_1'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
			df_val_ISIC2017['cell_type_task3_2'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
			df_val_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_binary, categories=self.classes_melanoma_binary).codes
			df_val_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
			df_val_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes

			df_test_ISIC2017['path'] = df_test_ISIC2017.image_id.map(imageid_path_test_dict_ISIC2017.get)
			df_test_ISIC2017['cell_type_binary'] = df_test_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
			df_test_ISIC2017['cell_type_task3_1'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
			df_test_ISIC2017['cell_type_task3_2'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
			df_test_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_binary, categories=self.classes_melanoma_binary).codes
			df_test_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
			df_test_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes



			logger.debug("Check null data in ISIC2017 training metadata")
			display(df_training_ISIC2017.isnull().sum())
			logger.debug("Check null data in ISIC2017 validation metadata")
			display(df_val_ISIC2017.isnull().sum())
			logger.debug("Check null data in ISIC2017 test metadata")
			display(df_test_ISIC2017.isnull().sum())


			# df_training_ISIC2017['ori_image'] = df_training_ISIC2017.path.map(
			# 	lambda x:(
			# 		img := Image.open(x), # [0]: PIL object
			# 		np.asarray(img), # [1]: pixel array
			# 	)
			# )
			
			df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			# df_val_ISIC2017['ori_image'] = df_val_ISIC2017.path.map(
			# 	lambda x:(
			# 		img := Image.open(x), # [0]: PIL object
			# 		np.asarray(img), # [1]: pixel array
			# 	)
			# )
			
			df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			# df_test_ISIC2017['ori_image'] = df_test_ISIC2017.path.map(
			# 	lambda x:(
			# 		img := Image.open(x), # [0]: PIL object
			# 		np.asarray(img), # [1]: pixel array
			# 	)
			# )
			
			df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_training_ISIC2017['img_sizes'] = df_training_ISIC2017.path.map(
				lambda x:(
					Image.open(x).size
				)
			)

			assert all(df_training_ISIC2017.cell_type_binary.unique() == df_test_ISIC2017.cell_type_binary.unique())
			assert all(df_val_ISIC2017.cell_type_binary.unique() == df_test_ISIC2017.cell_type_binary.unique())
			labels = df_training_ISIC2017.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)


			# df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

			# ISIC2017 datasets are divided into train/val/test already
			trainset_ISIC2017 = df_training_ISIC2017
			validationset_ISIC2017 = df_val_ISIC2017
			testset_ISIC2017 = df_test_ISIC2017

			preprocessor.saveNumpyImagesToFiles(trainset_ISIC2017, df_training_ISIC2017, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(validationset_ISIC2017, df_val_ISIC2017, val_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(testset_ISIC2017, df_test_ISIC2017, test_rgb_folder)

			# ISIC2017 binary images/labels
			trainpixels_ISIC2017 = list(map(lambda x:x[1], trainset_ISIC2017.image)) # Filter out only pixel from the list
			validationpixels_ISIC2017 = list(map(lambda x:x[1], validationset_ISIC2017.image)) # Filter out only pixel from the list
			testpixels_ISIC2017 = list(map(lambda x:x[1], testset_ISIC2017.image)) # Filter out only pixel from the list
			
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_ISIC2017 = preprocessor.normalizeImgs_ResNet50(trainpixels_ISIC2017)
				testimages_ISIC2017 = preprocessor.normalizeImgs_ResNet50(testpixels_ISIC2017)
				validationimages_ISIC2017 = preprocessor.normalizeImgs_ResNet50(validationpixels_ISIC2017)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_ISIC2017 = preprocessor.normalizeImgs_Xception(trainpixels_ISIC2017)
				testimages_ISIC2017 = preprocessor.normalizeImgs_Xception(testpixels_ISIC2017)
				validationimages_ISIC2017 = preprocessor.normalizeImgs_Xception(validationpixels_ISIC2017)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_ISIC2017 = preprocessor.normalizeImgs_inceptionV3(trainpixels_ISIC2017)
				testimages_ISIC2017 = preprocessor.normalizeImgs_inceptionV3(testpixels_ISIC2017)
				validationimages_ISIC2017 = preprocessor.normalizeImgs_inceptionV3(validationpixels_ISIC2017)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_ISIC2017 = preprocessor.normalizeImgs_vgg16(trainpixels_ISIC2017)
				testimages_ISIC2017 = preprocessor.normalizeImgs_vgg16(testpixels_ISIC2017)
				validationimages_ISIC2017 = preprocessor.normalizeImgs_vgg16(validationpixels_ISIC2017)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_ISIC2017 = preprocessor.normalizeImgs_vgg19(trainpixels_ISIC2017)
				testimages_ISIC2017 = preprocessor.normalizeImgs_vgg19(testpixels_ISIC2017)
				validationimages_ISIC2017 = preprocessor.normalizeImgs_vgg19(validationpixels_ISIC2017)
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_ISIC2017 = to_categorical(trainset_ISIC2017.cell_type_binary_idx, num_classes=2)
			testlabels_binary_ISIC2017 = to_categorical(testset_ISIC2017.cell_type_binary_idx, num_classes=2)
			validationlabels_binary_ISIC2017 = to_categorical(validationset_ISIC2017.cell_type_binary_idx, num_classes=2)

			assert num_train_img_ISIC2017 == len(trainpixels_ISIC2017)
			assert num_val_img_ISIC2017 == len(validationpixels_ISIC2017)
			assert num_test_img_ISIC2017 == len(testpixels_ISIC2017)
			assert len(trainpixels_ISIC2017) == trainlabels_binary_ISIC2017.shape[0]
			assert len(validationpixels_ISIC2017) == validationlabels_binary_ISIC2017.shape[0]
			assert len(testpixels_ISIC2017) == testlabels_binary_ISIC2017.shape[0]
			assert trainimages_ISIC2017.shape[0] == trainlabels_binary_ISIC2017.shape[0]
			assert validationimages_ISIC2017.shape[0] == validationlabels_binary_ISIC2017.shape[0]
			assert testimages_ISIC2017.shape[0] == testlabels_binary_ISIC2017.shape[0]
			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'ISIC2017'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_ISIC2017, testimages_ISIC2017, validationimages_ISIC2017,
				trainlabels_binary_ISIC2017, testlabels_binary_ISIC2017, validationlabels_binary_ISIC2017,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_ISIC2017_augmented, trainlabels_binary_ISIC2017_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_ISIC2017, trainlabels_binary_ISIC2017, \
						augment_ratio, df_training_ISIC2017)
				
				assert augmented_db_name.name == 'ISIC2017'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_ISIC2017_augmented, testimages_ISIC2017, validationimages_ISIC2017,
					trainlabels_binary_ISIC2017_augmented, testlabels_binary_ISIC2017, validationlabels_binary_ISIC2017,
					2), file_bin)
				file_bin.close()


		if datasettype.value == DatasetType.ISIC2018.value:
			ISIC2018_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC2018_Task3_Training_Input')
			ISIC2018_val_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC2018_Task3_Validation_Input')
			ISIC2018_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC2018_Task3_Test_Input')

			num_train_img_ISIC2018 = len(list(ISIC2018_training_path.glob('./*.jpg'))) # counts all ISIC2018 training images
			num_val_img_ISIC2018 = len(list(ISIC2018_val_path.glob('./*.jpg'))) # counts all ISIC2018 validation images
			num_test_img_ISIC2018 = len(list(ISIC2018_test_path.glob('./*.jpg'))) # counts all ISIC2018 test images

			assert num_train_img_ISIC2018 == 10015
			assert num_val_img_ISIC2018 == 193
			assert num_test_img_ISIC2018 == 1512

			logger.debug('%s %s', f"Images available in {datasettype.name} train dataset:", num_train_img_ISIC2018)
			logger.debug('%s %s', f"Images available in {datasettype.name} validation dataset:", num_val_img_ISIC2018)
			logger.debug('%s %s', f"Images available in {datasettype.name} test dataset:", num_test_img_ISIC2018)

			# ISIC2018: Dictionary for Image Names
			imageid_path_training_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_training_path, '*.jpg'))}
			imageid_path_val_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_val_path, '*.jpg'))}
			imageid_path_test_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_test_path, '*.jpg'))}

			
			# ISIC2018_columns = ['image_id', 'label']
			df_training_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(
				self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC2018_Task3_Training_GroundTruth', './ISIC2018_Task3_Training_GroundTruth.csv')),
				header=0)
			df_val_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(
				self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC2018_Task3_Validation_GroundTruth', './ISIC2018_Task3_Validation_GroundTruth.csv')),
				header=0)
			df_test_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(
				self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC2018_Task3_Test_GroundTruth', './ISIC2018_Task3_Test_GroundTruth.csv')),
				header=0)

			assert df_training_ISIC2018.shape[0] == 10015
			assert df_val_ISIC2018.shape[0] == 193
			assert df_test_ISIC2018.shape[0] == 1512

			logger.debug("Let's check ISIC2018 metadata briefly")
			logger.debug("This is ISIC2018 training data samples")
			display(df_training_ISIC2018.head())
			logger.debug("This is ISIC2018 validation data samples")
			display(df_val_ISIC2018.head())
			logger.debug("This is ISIC2018 test data samples")
			display(df_test_ISIC2018.head())



			# ISIC2018: Creating New Columns for better readability
			df_training_ISIC2018['path'] = df_training_ISIC2018['image'].map(imageid_path_training_dict_ISIC2018.get)
			df_training_ISIC2018['cell_type_binary'] = df_training_ISIC2018['MEL'].map(self.common_binary_label.get)
			df_training_ISIC2018['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2018.cell_type_binary, categories=self.classes_melanoma_binary).codes

			df_val_ISIC2018['path'] = df_val_ISIC2018['image'].map(imageid_path_val_dict_ISIC2018.get)
			df_val_ISIC2018['cell_type_binary'] = df_val_ISIC2018['MEL'].map(self.common_binary_label.get)
			df_val_ISIC2018['cell_type_binary_idx'] = pd.CategoricalIndex(df_val_ISIC2018.cell_type_binary, categories=self.classes_melanoma_binary).codes

			df_test_ISIC2018['path'] = df_test_ISIC2018['image'].map(imageid_path_test_dict_ISIC2018.get)
			df_test_ISIC2018['cell_type_binary'] = df_test_ISIC2018['MEL'].map(self.common_binary_label.get)
			df_test_ISIC2018['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2018.cell_type_binary, categories=self.classes_melanoma_binary).codes



			logger.debug("Check null data in ISIC2018 training metadata")
			display(df_training_ISIC2018.isnull().sum())
			logger.debug("Check null data in ISIC2018 validation metadata")
			display(df_val_ISIC2018.isnull().sum())
			logger.debug("Check null data in ISIC2018 test metadata")
			display(df_test_ISIC2018.isnull().sum())
			
			df_training_ISIC2018['image'] = df_training_ISIC2018.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)
			
			df_val_ISIC2018['image'] = df_val_ISIC2018.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			
			df_test_ISIC2018['image'] = df_test_ISIC2018.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_training_ISIC2018['img_sizes'] = df_training_ISIC2018.path.map(
				lambda x:(
					Image.open(x).size
				)
			)

			assert all(df_training_ISIC2018.cell_type_binary.unique() == df_test_ISIC2018.cell_type_binary.unique())
			assert all(df_val_ISIC2018.cell_type_binary.unique() == df_test_ISIC2018.cell_type_binary.unique())
			labels = df_training_ISIC2018.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)


			# df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

			# ISIC2018 datasets are divided into train/val/test already
			trainset_ISIC2018 = df_training_ISIC2018
			validationset_ISIC2018 = df_val_ISIC2018
			testset_ISIC2018 = df_test_ISIC2018

			preprocessor.saveNumpyImagesToFiles(trainset_ISIC2018, df_training_ISIC2018, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(validationset_ISIC2018, df_val_ISIC2018, val_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(testset_ISIC2018, df_test_ISIC2018, test_rgb_folder)

			# ISIC2018 binary images/labels
			trainpixels_ISIC2018 = list(map(lambda x:x[1], trainset_ISIC2018.image)) # Filter out only pixel from the list
			validationpixels_ISIC2018 = list(map(lambda x:x[1], validationset_ISIC2018.image)) # Filter out only pixel from the list
			testpixels_ISIC2018 = list(map(lambda x:x[1], testset_ISIC2018.image)) # Filter out only pixel from the list
			
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_ISIC2018 = preprocessor.normalizeImgs_ResNet50(trainpixels_ISIC2018)
				testimages_ISIC2018 = preprocessor.normalizeImgs_ResNet50(testpixels_ISIC2018)
				validationimages_ISIC2018 = preprocessor.normalizeImgs_ResNet50(validationpixels_ISIC2018)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_ISIC2018 = preprocessor.normalizeImgs_Xception(trainpixels_ISIC2018)
				testimages_ISIC2018 = preprocessor.normalizeImgs_Xception(testpixels_ISIC2018)
				validationimages_ISIC2018 = preprocessor.normalizeImgs_Xception(validationpixels_ISIC2018)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_ISIC2018 = preprocessor.normalizeImgs_inceptionV3(trainpixels_ISIC2018)
				testimages_ISIC2018 = preprocessor.normalizeImgs_inceptionV3(testpixels_ISIC2018)
				validationimages_ISIC2018 = preprocessor.normalizeImgs_inceptionV3(validationpixels_ISIC2018)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_ISIC2018 = preprocessor.normalizeImgs_vgg16(trainpixels_ISIC2018)
				testimages_ISIC2018 = preprocessor.normalizeImgs_vgg16(testpixels_ISIC2018)
				validationimages_ISIC2018 = preprocessor.normalizeImgs_vgg16(validationpixels_ISIC2018)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_ISIC2018 = preprocessor.normalizeImgs_vgg19(trainpixels_ISIC2018)
				testimages_ISIC2018 = preprocessor.normalizeImgs_vgg19(testpixels_ISIC2018)
				validationimages_ISIC2018 = preprocessor.normalizeImgs_vgg19(validationpixels_ISIC2018)
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_ISIC2018 = to_categorical(trainset_ISIC2018.cell_type_binary_idx, num_classes=2)
			testlabels_binary_ISIC2018 = to_categorical(testset_ISIC2018.cell_type_binary_idx, num_classes=2)
			validationlabels_binary_ISIC2018 = to_categorical(validationset_ISIC2018.cell_type_binary_idx, num_classes=2)

			assert num_train_img_ISIC2018 == len(trainpixels_ISIC2018)
			assert num_val_img_ISIC2018 == len(validationpixels_ISIC2018)
			assert num_test_img_ISIC2018 == len(testpixels_ISIC2018)
			assert len(trainpixels_ISIC2018) == trainlabels_binary_ISIC2018.shape[0]
			assert len(validationpixels_ISIC2018) == validationlabels_binary_ISIC2018.shape[0]
			assert len(testpixels_ISIC2018) == testlabels_binary_ISIC2018.shape[0]
			assert trainimages_ISIC2018.shape[0] == trainlabels_binary_ISIC2018.shape[0]
			assert validationimages_ISIC2018.shape[0] == validationlabels_binary_ISIC2018.shape[0]
			assert testimages_ISIC2018.shape[0] == testlabels_binary_ISIC2018.shape[0]
			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'ISIC2018'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_ISIC2018, testimages_ISIC2018, validationimages_ISIC2018,
				trainlabels_binary_ISIC2018, testlabels_binary_ISIC2018, validationlabels_binary_ISIC2018,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_ISIC2018_augmented, trainlabels_binary_ISIC2018_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_ISIC2018, trainlabels_binary_ISIC2018, \
						augment_ratio, df_training_ISIC2018)
				
				assert augmented_db_name.name == 'ISIC2018'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_ISIC2018_augmented, testimages_ISIC2018, validationimages_ISIC2018,
					trainlabels_binary_ISIC2018_augmented, testlabels_binary_ISIC2018, validationlabels_binary_ISIC2018,
					2), file_bin)
				file_bin.close()

		if datasettype.value == DatasetType.ISIC2019.value:
			ISIC2019_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC_2019_Training_Input')

			num_train_img_ISIC2019 = len(list(ISIC2019_training_path.glob('./*.jpg'))) # counts all ISIC2019 training images

			assert num_train_img_ISIC2019 == 25331

			logger.debug('%s %s', f"Images available in {datasettype.name} train dataset:", num_train_img_ISIC2019)

			# ISIC2019: Dictionary for Image Names
			imageid_path_training_dict_ISIC2019 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2019_training_path, '*.jpg'))}

			
			# ISIC2018_columns = ['image_id', 'label']
			df_training_ISIC2019 = pd.read_csv(str(pathlib.Path.joinpath(
				self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC_2019_Training_GroundTruth.csv')),
				header=0)

			assert df_training_ISIC2019.shape[0] == 25331
			

			logger.debug("Let's check ISIC2019 metadata briefly")
			logger.debug("This is ISIC2019 training data samples")
			display(df_training_ISIC2019.head())



			# ISIC2019: Creating New Columns for better readability
			df_training_ISIC2019['path'] = df_training_ISIC2019['image'].map(imageid_path_training_dict_ISIC2019.get)
			df_training_ISIC2019['cell_type_binary'] = df_training_ISIC2019['MEL'].map(self.common_binary_label.get)
			df_training_ISIC2019['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2019.cell_type_binary, categories=self.classes_melanoma_binary).codes


			logger.debug("Check null data in ISIC2019 training metadata")
			display(df_training_ISIC2019.isnull().sum())
			
			df_training_ISIC2019['image'] = df_training_ISIC2019.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_training_ISIC2019['img_sizes'] = df_training_ISIC2019.path.map(
				lambda x:(
					Image.open(x).size
				)
			)



			# assert all(df_training_ISIC2019.cell_type_binary.unique() == df_test_ISIC2019.cell_type_binary.unique())
			# assert all(df_val_ISIC2019.cell_type_binary.unique() == df_test_ISIC2019.cell_type_binary.unique())
			labels = df_training_ISIC2019.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)

			# Dividing ISIC2019 into train/val set
			trainset_ISIC2019, validationset_ISIC2019 = train_test_split(df_training_ISIC2019, test_size=0.2,random_state = 1)

			preprocessor.saveNumpyImagesToFiles(trainset_ISIC2019, df_training_ISIC2019, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(validationset_ISIC2019, df_training_ISIC2019, train_rgb_folder)

			# ISIC2019 binary images/labels
			trainpixels_ISIC2019 = list(map(lambda x:x[1], trainset_ISIC2019.image)) # Filter out only pixel from the list
			validationpixels_ISIC2019 = list(map(lambda x:x[1], validationset_ISIC2019.image)) # Filter out only pixel from the list
			
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_ISIC2019 = preprocessor.normalizeImgs_ResNet50(trainpixels_ISIC2019)
				validationimages_ISIC2019 = preprocessor.normalizeImgs_ResNet50(validationpixels_ISIC2019)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_ISIC2019 = preprocessor.normalizeImgs_Xception(trainpixels_ISIC2019)
				validationimages_ISIC2019 = preprocessor.normalizeImgs_Xception(validationpixels_ISIC2019)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_ISIC2019 = preprocessor.normalizeImgs_inceptionV3(trainpixels_ISIC2019)
				validationimages_ISIC2019 = preprocessor.normalizeImgs_inceptionV3(validationpixels_ISIC2019)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_ISIC2019 = preprocessor.normalizeImgs_vgg16(trainpixels_ISIC2019)
				validationimages_ISIC2019 = preprocessor.normalizeImgs_vgg16(validationpixels_ISIC2019)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_ISIC2019 = preprocessor.normalizeImgs_vgg19(trainpixels_ISIC2019)
				validationimages_ISIC2019 = preprocessor.normalizeImgs_vgg19(validationpixels_ISIC2019)

			trainlabels_binary_ISIC2019 = to_categorical(trainset_ISIC2019.cell_type_binary_idx, num_classes=2)
			validationlabels_binary_ISIC2019 = to_categorical(validationset_ISIC2019.cell_type_binary_idx, num_classes=2)

			assert num_train_img_ISIC2019 == len(trainpixels_ISIC2019) + len(validationpixels_ISIC2019)
			assert len(trainpixels_ISIC2019) == trainlabels_binary_ISIC2019.shape[0]
			assert len(validationpixels_ISIC2019) == validationlabels_binary_ISIC2019.shape[0]
			assert trainimages_ISIC2019.shape[0] == trainlabels_binary_ISIC2019.shape[0]
			assert validationimages_ISIC2019.shape[0] == validationlabels_binary_ISIC2019.shape[0]

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'ISIC2019'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_ISIC2019, None, validationimages_ISIC2019,
				trainlabels_binary_ISIC2019, None, validationlabels_binary_ISIC2019,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_ISIC2019_augmented, trainlabels_binary_ISIC2019_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_ISIC2019, trainlabels_binary_ISIC2019, \
						augment_ratio, df_training_ISIC2019)
				
				assert augmented_db_name.name == 'ISIC2019'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_ISIC2019_augmented, None, validationimages_ISIC2019,
					trainlabels_binary_ISIC2019_augmented, None, validationlabels_binary_ISIC2019,
					2), file_bin)
				file_bin.close()

		if datasettype.value == DatasetType.ISIC2020.value:
			ISIC2020_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasettype.name}', './train')

			num_train_img_ISIC2020 = len(list(ISIC2020_training_path.glob('./*.jpg'))) # counts all ISIC2019 training images

			assert num_train_img_ISIC2020 == 33126

			logger.debug('%s %s', f"Images available in {datasettype.name} train dataset:", num_train_img_ISIC2020)

			# ISIC2020: Dictionary for Image Names
			imageid_path_training_dict_ISIC2020 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2020_training_path, '*.jpg'))}

			
			# ISIC2018_columns = ['image_id', 'label']
			df_training_ISIC2020 = pd.read_csv(str(pathlib.Path.joinpath(
				self.base_dir, './melanomaDB', f'./{datasettype.name}', './ISIC_2020_Training_GroundTruth.csv')),
				header=0)

			assert df_training_ISIC2020.shape[0] == 33126
			

			logger.debug("Let's check ISIC2020 metadata briefly")
			logger.debug("This is ISIC2020 training data samples")
			display(df_training_ISIC2020.head())



			# ISIC2020: Creating New Columns for better readability
			df_training_ISIC2020['path'] = df_training_ISIC2020['image_name'].map(imageid_path_training_dict_ISIC2020.get)
			df_training_ISIC2020['cell_type_binary'] = df_training_ISIC2020['benign_malignant'].map(self.lesion_type_binary_dict_training_ISIC2020.get)
			df_training_ISIC2020['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2020.cell_type_binary, categories=self.classes_melanoma_binary).codes


			logger.debug("Check null data in ISIC2020 training metadata")
			display(df_training_ISIC2020.isnull().sum())
			
			df_training_ISIC2020['image'] = df_training_ISIC2020.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_training_ISIC2020['img_sizes'] = df_training_ISIC2020.path.map(
				lambda x:(
					Image.open(x).size
				)
			)



			# assert all(df_training_ISIC2020.cell_type_binary.unique() == df_test_ISIC2020.cell_type_binary.unique())
			# assert all(df_val_ISIC2020.cell_type_binary.unique() == df_test_ISIC2020.cell_type_binary.unique())
			labels = df_training_ISIC2020.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)


			# df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

			# Dividing ISIC2020 into train/val set
			trainset_ISIC2020, validationset_ISIC2020 = train_test_split(df_training_ISIC2020, test_size=0.2,random_state = 1)

			preprocessor.saveNumpyImagesToFiles(trainset_ISIC2020, df_training_ISIC2020, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(validationset_ISIC2020, df_training_ISIC2020, train_rgb_folder)

			# ISIC2020 binary images/labels
			trainpixels_ISIC2020 = list(map(lambda x:x[1], trainset_ISIC2020.image)) # Filter out only pixel from the list
			validationpixels_ISIC2020 = list(map(lambda x:x[1], validationset_ISIC2020.image)) # Filter out only pixel from the list
			
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_ISIC2020 = preprocessor.normalizeImgs_ResNet50(trainpixels_ISIC2020)
				validationimages_ISIC2020 = preprocessor.normalizeImgs_ResNet50(validationpixels_ISIC2020)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_ISIC2020 = preprocessor.normalizeImgs_Xception(trainpixels_ISIC2020)
				validationimages_ISIC2020 = preprocessor.normalizeImgs_Xception(validationpixels_ISIC2020)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_ISIC2020 = preprocessor.normalizeImgs_inceptionV3(trainpixels_ISIC2020)
				validationimages_ISIC2020 = preprocessor.normalizeImgs_inceptionV3(validationpixels_ISIC2020)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_ISIC2020 = preprocessor.normalizeImgs_vgg16(trainpixels_ISIC2020)
				validationimages_ISIC2020 = preprocessor.normalizeImgs_vgg16(validationpixels_ISIC2020)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_ISIC2020 = preprocessor.normalizeImgs_vgg19(trainpixels_ISIC2020)
				validationimages_ISIC2020 = preprocessor.normalizeImgs_vgg19(validationpixels_ISIC2020)
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_ISIC2020 = to_categorical(trainset_ISIC2020.cell_type_binary_idx, num_classes=2)
			validationlabels_binary_ISIC2020 = to_categorical(validationset_ISIC2020.cell_type_binary_idx, num_classes=2)

			assert num_train_img_ISIC2020 == len(trainpixels_ISIC2020) + len(validationpixels_ISIC2020)
			assert len(trainpixels_ISIC2020) == trainlabels_binary_ISIC2020.shape[0]
			assert len(validationpixels_ISIC2020) == validationlabels_binary_ISIC2020.shape[0]
			assert trainimages_ISIC2020.shape[0] == trainlabels_binary_ISIC2020.shape[0]
			assert validationimages_ISIC2020.shape[0] == validationlabels_binary_ISIC2020.shape[0]

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'ISIC2020'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_ISIC2020, None, validationimages_ISIC2020,
				trainlabels_binary_ISIC2020, None, validationlabels_binary_ISIC2020,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_ISIC2020_augmented, trainlabels_binary_ISIC2020_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_ISIC2020, trainlabels_binary_ISIC2020, \
						augment_ratio, df_training_ISIC2020)
				
				assert augmented_db_name.name == 'ISIC2020'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_ISIC2020_augmented, None, validationimages_ISIC2020,
					trainlabels_binary_ISIC2020_augmented, None, validationlabels_binary_ISIC2020,
					2), file_bin)
				file_bin.close()

		if datasettype.value == DatasetType.PH2.value:
			PH2path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './PH2Dataset')

			img_path =pathlib.Path.joinpath(PH2path, './PH2 Dataset images')

			num_imgs = len(list(img_path.glob('*/*_Dermoscopic_Image/*.bmp'))) # counts all PH2 training images

			assert num_imgs == 200

			logger.debug('%s %s', f"Images available in {datasettype.name} dataset:", num_imgs)

			imageid_path_dict_PH2 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(img_path, '*/*_Dermoscopic_Image/*.bmp'))}

			
			df_PH2 = pd.read_excel(str(PH2path) + '/PH2_dataset.xlsx', header=12)

			assert df_PH2.shape[0] == 200

			logger.debug("Let's check PH2 metadata briefly")
			logger.debug("This is PH2 data samples")
			display(df_PH2.head())



			# PH2: Creating New Columns for better readability
			df_PH2['path'] = df_PH2['Image Name'].map(imageid_path_dict_PH2.get)
			df_PH2['cell_type_binary'] = np.where(df_PH2['Melanoma'] == 'X', 'Melanoma', 'Non-Melanoma')
			df_PH2['cell_type_binary_idx'] = pd.CategoricalIndex(df_PH2.cell_type_binary, categories=self.classes_melanoma_binary).codes


			logger.debug("Check null data in ISIC2020 training metadata")
			display(df_PH2.isnull().sum())
			
			df_PH2['image'] = df_PH2.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			labels = df_PH2.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)


			# df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

			# Dividing PH2 into train/val set
			# trainset_ISIC2020, validationset_ISIC2020 = train_test_split(df_training_ISIC2020, test_size=0.2,random_state = 1)
			

			preprocessor.saveNumpyImagesToFiles(df_PH2, df_PH2, train_rgb_folder)

			# PH2 binary images/labels
			trainpixels_PH2 = list(map(lambda x:x[1], df_PH2.image)) # Filter out only pixel from the list
			
			
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_PH2 = preprocessor.normalizeImgs_ResNet50(trainpixels_PH2)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_PH2 = preprocessor.normalizeImgs_Xception(trainpixels_PH2)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_PH2 = preprocessor.normalizeImgs_inceptionV3(trainpixels_PH2)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_PH2 = preprocessor.normalizeImgs_vgg16(trainpixels_PH2)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_PH2 = preprocessor.normalizeImgs_vgg19(trainpixels_PH2)
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_PH2 = to_categorical(df_PH2.cell_type_binary_idx, num_classes=2)

			assert num_imgs == len(trainpixels_PH2)
			assert len(trainpixels_PH2) == trainlabels_binary_PH2.shape[0]
			assert trainimages_PH2.shape[0] == trainlabels_binary_PH2.shape[0]

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'PH2'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_PH2, None, None,
				trainlabels_binary_PH2, None, None,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_PH2_augmented, trainlabels_binary_PH2_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_PH2, trainlabels_binary_PH2, \
						augment_ratio, df_PH2)
				
				assert augmented_db_name.name == 'PH2'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_PH2_augmented, None, None,
					trainlabels_binary_PH2_augmented, None, None,
					2), file_bin)
				file_bin.close()


		if datasettype.value == DatasetType._7_point_criteria.value:
			_7pointdb_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './release_v0')

			img_path =pathlib.Path.joinpath(_7pointdb_path, './images')

			num_imgs = len(list(img_path.glob('*/*.*'))) # counts all 7-point db training images

			assert num_imgs == 2013 # Num of files in folder

			logger.debug('%s %s', f"Images available in {datasettype.name} dataset:", num_imgs)


			imagedir_dict = {os.path.join(os.path.basename(os.path.dirname(x)), os.path.basename(x)): x for x in glob(os.path.join(img_path, '*/*.*'))}
			imagedir_dict_lower = {k.lower(): v for k, v in imagedir_dict.items()}
			# imagedir_dict_lower = list(map(lambda x: x.lower(), list(imagedir_dict.keys())))
			# imageid_path_dict_7pointdb = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(img_path, '*/*.*'))}

			# imagedir_dict_lower_dict = dict()
			# for ele in imagedir_dict_lower:
			# 	imagedir_dict_lower_dict[str(ele)] = ele
			
			df_7pointdb = pd.read_csv(str(_7pointdb_path) + '/meta/meta.csv', header=0)

			assert df_7pointdb.shape[0] == 1011 # meta rows

			logger.debug("Let's check 7-point criteria db metadata briefly")
			logger.debug("This is 7-point criteria db samples")
			display(df_7pointdb.head())

			# 7 point criteria db: Creating New Columns for better readability
			# df_7pointdb['path_clinic'] = df_7pointdb['clinic'].str.lower().map(imagedir_dict_lower.get)
			df_7pointdb['path'] = df_7pointdb['derm'].str.lower().map(imagedir_dict_lower.get)
			# df_7pointdb['path_clinic'].shape[0] == 1011
			df_7pointdb['path'].shape[0] == 1011
			df_7pointdb['cell_type_binary'] = df_7pointdb['diagnosis'].apply(lambda x: 'Melanoma' if 'melanoma' in x else 'Non-Melanoma')
			df_7pointdb['cell_type_binary_idx'] = pd.CategoricalIndex(df_7pointdb.cell_type_binary, categories=self.classes_melanoma_binary).codes


			logger.debug("Check null data in 7 point db training metadata")
			display(df_7pointdb.isnull().sum())
			
			# df_7pointdb['image_clinic'] = df_7pointdb.path_clinic.map(
			# 	lambda x:(
			# 		# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
			# 		img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
			# 		# np.asarray(img), # [1]: pixel array
			# 		img_to_array(img), # [1]: pixel array
			# 		currentPath := pathlib.Path(x), # [2]: PosixPath
			# 		# img.save(f"{whole_rgb_folder}/{currentPath.name}")
			# 	)
			# )
			df_7pointdb['image'] = df_7pointdb.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			labels = df_7pointdb.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_feature_folder}/{i}", exist_ok=True)

			df_training_index = pd.read_csv(str(_7pointdb_path) + '/meta/train_indexes.csv', header=0)
			df_validation_index = pd.read_csv(str(_7pointdb_path) + '/meta/valid_indexes.csv', header=0)
			df_test_index = pd.read_csv(str(_7pointdb_path) + '/meta/test_indexes.csv', header=0)
			# df_training_7pointdb = df_7pointdb[df_7pointdb.index.isin(df_training_index['indexes'])]
			df_training_7pointdb = df_7pointdb.filter(items = df_training_index['indexes'], axis=0)
			df_validation_7pointdb = df_7pointdb.filter(items = df_validation_index['indexes'], axis=0)
			df_test_7pointdb = df_7pointdb.filter(items = df_test_index['indexes'], axis=0)
			df_training_7pointdb.shape[0] == 413
			df_validation_7pointdb.shape[0] == 203
			df_test_7pointdb.shape[0] == 395
			

			# df_training_7pointdb['image'] = df_training_7pointdb.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))			

			preprocessor.saveNumpyImagesToFiles(df_training_7pointdb, df_7pointdb, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(df_validation_7pointdb, df_7pointdb, val_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(df_test_7pointdb, df_7pointdb, test_rgb_folder)

			# 7 point db binary images/labels
			trainpixels_7pointdb = list(map(lambda x:x[1], df_training_7pointdb.image)) # Filter out only pixel from the list
			validationpixels_7pointdb = list(map(lambda x:x[1], df_validation_7pointdb.image)) # Filter out only pixel from the list
			testpixels_7pointdb = list(map(lambda x:x[1], df_test_7pointdb.image)) # Filter out only pixel from the list
			
			
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_7pointdb = preprocessor.normalizeImgs_ResNet50(trainpixels_7pointdb)
				validationimages_7pointdb = preprocessor.normalizeImgs_ResNet50(validationpixels_7pointdb)
				testimages_7pointdb = preprocessor.normalizeImgs_ResNet50(testpixels_7pointdb)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_7pointdb = preprocessor.normalizeImgs_Xception(trainpixels_7pointdb)
				validationimages_7pointdb = preprocessor.normalizeImgs_Xception(validationpixels_7pointdb)
				testimages_7pointdb = preprocessor.normalizeImgs_Xception(testpixels_7pointdb)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_7pointdb = preprocessor.normalizeImgs_inceptionV3(trainpixels_7pointdb)
				validationimages_7pointdb = preprocessor.normalizeImgs_inceptionV3(validationpixels_7pointdb)
				testimages_7pointdb = preprocessor.normalizeImgs_inceptionV3(testpixels_7pointdb)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_7pointdb = preprocessor.normalizeImgs_vgg16(trainpixels_7pointdb)
				validationimages_7pointdb = preprocessor.normalizeImgs_vgg16(validationpixels_7pointdb)
				testimages_7pointdb = preprocessor.normalizeImgs_vgg16(testpixels_7pointdb)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_7pointdb = preprocessor.normalizeImgs_vgg19(trainpixels_7pointdb)
				validationimages_7pointdb = preprocessor.normalizeImgs_vgg19(validationpixels_7pointdb)
				testimages_7pointdb = preprocessor.normalizeImgs_vgg19(testpixels_7pointdb)
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_7pointdb = to_categorical(df_training_7pointdb.cell_type_binary_idx, num_classes=2)
			validationlabels_binary_7pointdb = to_categorical(df_validation_7pointdb.cell_type_binary_idx, num_classes=2)
			testlabels_binary_7pointdb = to_categorical(df_test_7pointdb.cell_type_binary_idx, num_classes=2)

			
			assert len(trainpixels_7pointdb) == trainlabels_binary_7pointdb.shape[0]
			assert len(validationpixels_7pointdb) == validationlabels_binary_7pointdb.shape[0]
			assert len(testpixels_7pointdb) == testlabels_binary_7pointdb.shape[0]
			assert trainimages_7pointdb.shape[0] == trainlabels_binary_7pointdb.shape[0]
			assert validationimages_7pointdb.shape[0] == validationlabels_binary_7pointdb.shape[0]
			assert testimages_7pointdb.shape[0] == testlabels_binary_7pointdb.shape[0]

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == '_7_point_criteria'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_7pointdb, testimages_7pointdb, validationimages_7pointdb,
				trainlabels_binary_7pointdb, testlabels_binary_7pointdb, validationlabels_binary_7pointdb,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_7pointdb_augmented, trainlabels_binary_7pointdb_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_7pointdb, trainlabels_binary_7pointdb, \
						augment_ratio, df_training_7pointdb)
				
				assert augmented_db_name.name == '_7_point_criteria'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_7pointdb_augmented, testimages_7pointdb, validationimages_7pointdb,
					trainlabels_binary_7pointdb_augmented, testlabels_binary_7pointdb, validationlabels_binary_7pointdb,
					2), file_bin)
				file_bin.close()

		if datasettype.value == DatasetType.PAD_UFES_20.value:
			dbpath = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './PAD-UFES-20')

			img_path =pathlib.Path.joinpath(dbpath, './images')

			num_imgs = len(list(img_path.glob('imgs_part_*/*.*'))) # counts all PAD_UFES_20 training images

			assert num_imgs == 2298

			logger.debug('%s %s', f"Images available in {datasettype.name} dataset:", num_imgs)

			imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(img_path, 'imgs_part_*/*.*'))}

			
			df_PAD_UFES_20 = pd.read_csv(str(dbpath) + '/metadata.csv', header=0)

			assert df_PAD_UFES_20.shape[0] == 2298

			logger.debug("Let's check PAD UFES 20 metadata briefly")
			logger.debug("This is PAD UFES 20 data samples")
			display(df_PAD_UFES_20.head())



			# PAD UFES 20: Creating New Columns for better readability
			df_PAD_UFES_20['path'] = df_PAD_UFES_20['img_id'].map(imageid_path_dict.get)
			df_PAD_UFES_20['cell_type_binary'] = np.where(df_PAD_UFES_20['diagnostic'] == 'MEL', 'Melanoma', 'Non-Melanoma')
			df_PAD_UFES_20['cell_type_binary_idx'] = pd.CategoricalIndex(df_PAD_UFES_20.cell_type_binary, categories=self.classes_melanoma_binary).codes


			logger.debug("Check null data in PAD UFES 20 training metadata")
			display(df_PAD_UFES_20.isnull().sum())
			
			df_PAD_UFES_20['image'] = df_PAD_UFES_20.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_PAD_UFES_20['img_sizes'] = df_PAD_UFES_20.path.map(
				lambda x:(
					Image.open(x).size
				)
			)
			# max_height, max_width = df_PAD_UFES_20['img_sizes'].max()
			# min_height, min_width = df_PAD_UFES_20['img_sizes'].min()
			# max_min = dict(max_height=max_height, min_height=min_height, max_width=max_width, min_width=min_width)

			


			

			labels = df_PAD_UFES_20.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)


			# df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
			# df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

			# Dividing PAD UFES 20 into train/val set
			trainset_PAD_UFES_20, validationset_PAD_UFES_20 = train_test_split(df_PAD_UFES_20, test_size=0.2,random_state = 1)
			

			preprocessor.saveNumpyImagesToFiles(df_PAD_UFES_20, df_PAD_UFES_20, train_rgb_folder)

			# PAD UFES 20 binary images/labels
			trainpixels_PAD_UFES_20 = list(map(lambda x:x[1], trainset_PAD_UFES_20.image)) # Filter out only pixel from the list
			validationpixels_PAD_UFES_20 = list(map(lambda x:x[1], validationset_PAD_UFES_20.image)) # Filter out only pixel from the list
			
			
			if networktype.name == NetworkType.ResNet50.name:
				trainimages_PAD_UFES_20 = preprocessor.normalizeImgs_ResNet50(trainpixels_PAD_UFES_20)
				validationimages_PAD_UFES_20 = preprocessor.normalizeImgs_ResNet50(validationpixels_PAD_UFES_20)
			elif networktype.name == NetworkType.Xception.name:
				trainimages_PAD_UFES_20 = preprocessor.normalizeImgs_Xception(trainpixels_PAD_UFES_20)
				validationimages_PAD_UFES_20 = preprocessor.normalizeImgs_Xception(validationpixels_PAD_UFES_20)
			elif networktype.name == NetworkType.InceptionV3.name:
				trainimages_PAD_UFES_20 = preprocessor.normalizeImgs_inceptionV3(trainpixels_PAD_UFES_20)
				validationimages_PAD_UFES_20 = preprocessor.normalizeImgs_inceptionV3(validationpixels_PAD_UFES_20)
			elif networktype.name == NetworkType.VGG16.name:
				trainimages_PAD_UFES_20 = preprocessor.normalizeImgs_vgg16(trainpixels_PAD_UFES_20)
				validationimages_PAD_UFES_20 = preprocessor.normalizeImgs_vgg16(validationpixels_PAD_UFES_20)
			elif networktype.name == NetworkType.VGG19.name:
				trainimages_PAD_UFES_20 = preprocessor.normalizeImgs_vgg19(trainpixels_PAD_UFES_20)
				validationimages_PAD_UFES_20 = preprocessor.normalizeImgs_vgg19(validationpixels_PAD_UFES_20)
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary_PAD_UFES_20 = to_categorical(trainset_PAD_UFES_20.cell_type_binary_idx, num_classes=2)
			validationlabels_binary_PAD_UFES_20 = to_categorical(validationset_PAD_UFES_20.cell_type_binary_idx, num_classes=2)

			assert num_imgs == len(trainpixels_PAD_UFES_20) + len(validationpixels_PAD_UFES_20)
			assert len(trainpixels_PAD_UFES_20) == trainlabels_binary_PAD_UFES_20.shape[0]
			assert len(validationpixels_PAD_UFES_20) == validationlabels_binary_PAD_UFES_20.shape[0]
			assert trainimages_PAD_UFES_20.shape[0] == trainlabels_binary_PAD_UFES_20.shape[0]
			assert validationimages_PAD_UFES_20.shape[0] == validationlabels_binary_PAD_UFES_20.shape[0]

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'PAD_UFES_20'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages_PAD_UFES_20, None, validationimages_PAD_UFES_20,
				trainlabels_binary_PAD_UFES_20, None, validationlabels_binary_PAD_UFES_20,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_PAD_UFES_20_augmented, trainlabels_binary_PAD_UFES_20_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages_PAD_UFES_20, trainlabels_binary_PAD_UFES_20, \
						augment_ratio, trainset_PAD_UFES_20)
				
				assert augmented_db_name.name == 'PAD_UFES_20'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_PAD_UFES_20_augmented, None, validationimages_PAD_UFES_20,
					trainlabels_binary_PAD_UFES_20_augmented, None, validationlabels_binary_PAD_UFES_20,
					2), file_bin)
				file_bin.close()

		if datasettype.value == DatasetType.KaggleMB.value:

			dbpath = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './Kaggle_malignant_benign_DB')

			num_imgs = len(list(dbpath.glob('t*/*/*.*'))) # counts all Kaggle Malignant Benign training images

			# train: 1440 benign, 1197 malignant; test: 360 benign + 300 malignant
			assert num_imgs == 1440+1197+360+300

			logger.debug('%s %s', f"Images available in {datasettype.name} dataset:", num_imgs)

			# imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(dbpath, 't*/*/*.*'))}
			paths = glob(os.path.join(dbpath, 't*/*/*.*'))
			# labels_dict = {os.path.basename(x): x for x in os.path.abspath(os.path.join(os.path.join(imageid_path_dict.values()), os.pardir))}
			df = pd.DataFrame()


			# Kaggle MB: Creating New Columns for better readability
			df['path'] = paths
			df['label'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir))))
			df['portion'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir, os.pardir))))
			assert df['label'].unique().shape[0] == 2
			df['cell_type_binary'] = np.where(df['label'] == 'malignant', 'Melanoma', 'Non-Melanoma')
			df['cell_type_binary_idx'] = pd.CategoricalIndex(df.cell_type_binary, categories=self.classes_melanoma_binary).codes


			logger.debug("Check null data in Kaggle MB training metadata")
			display(df.isnull().sum())
			
			df['image'] = df.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df['img_sizes'] = df.path.map(
				lambda x:(
					Image.open(x).size
				)
			)

			labels = df.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
					
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)

			# Dividing Kaggle MB into train/test set
			df_trainset_temp = df.query('portion == "train"')
			df_testset = df.query('portion == "test"')

			# Dividing PAD UFES 20 into train/val set
			df_trainset, df_validationset = train_test_split(df_trainset_temp, test_size=0.2,random_state = 1)
			

			preprocessor.saveNumpyImagesToFiles(df_trainset, df, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(df_validationset, df, val_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(df_testset, df, test_rgb_folder)

			# KaggleMB binary images/labels
			trainpixels = list(map(lambda x:x[1], df_trainset.image)) # Filter out only pixel from the list
			validationpixels = list(map(lambda x:x[1], df_validationset.image)) # Filter out only pixel from the list
			testpixels = list(map(lambda x:x[1], df_testset.image)) # Filter out only pixel from the list
			
			trainimages = preprocessor.normalizeImgs(trainpixels, networktype)
			validationimages = preprocessor.normalizeImgs(validationpixels, networktype)
			testimages = preprocessor.normalizeImgs(testpixels, networktype)
			
			
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary = to_categorical(df_trainset.cell_type_binary_idx, num_classes=2)
			validationlabels_binary = to_categorical(df_validationset.cell_type_binary_idx, num_classes=2)
			testlabels_binary = to_categorical(df_testset.cell_type_binary_idx, num_classes=2)

			assert len(trainpixels)+len(validationpixels) == 1440+1197
			assert len(testpixels) == 360+300
			assert len(trainpixels) == trainlabels_binary.shape[0]
			assert len(validationpixels) == validationlabels_binary.shape[0]
			assert len(testpixels) == testlabels_binary.shape[0]
			assert trainimages.shape[0] == trainlabels_binary.shape[0]
			assert validationimages.shape[0] == validationlabels_binary.shape[0]
			assert testimages.shape[0] == testlabels_binary.shape[0]

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'KaggleMB'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages, testimages, validationimages,
				trainlabels_binary, testlabels_binary, validationlabels_binary,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_augmented, trainlabels_binary_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages, trainlabels_binary, \
						augment_ratio, df_trainset)
				
				assert augmented_db_name.name == 'KaggleMB'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_augmented, testimages, validationimages,
					trainlabels_binary_augmented, testlabels_binary, validationlabels_binary,
					2), file_bin)
				file_bin.close()

			# rootpath = f'/hpcstor6/scratch01/s/sanghyuk.kim001'
			# directoryPath = rootpath + '/melanomaDB/Kaggle_malignant_benign_DB'

			# labels = os.listdir(directoryPath+'/train')

			# assert labels == os.listdir(directoryPath+'/test')

			# if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
			# 	for i in labels:
			# 		os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
			# 		os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
			# 		os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
			# 		os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
			
			

			# path_benign_train = f'{directoryPath}/train/benign'
			# path_malignant_train = f'{directoryPath}/train/malignant'
			# path_benign_val = None
			# path_malignant_val = None
			# path_benign_test = f'{directoryPath}/test/benign'
			# path_malignant_test = f'{directoryPath}/test/malignant'

			# debug_paths = {"whole_rgb_folder": whole_rgb_folder, "train_rgb_folder": train_rgb_folder, "val_rgb_folder": val_rgb_folder, "test_rgb_folder": test_rgb_folder}

			# new_directory = str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './customDB', f'./{networktype.name}/'))
			# new_filename = f'{datasettype.name}_{img_height}h_{img_width}w.pkl'
			# self.saveDatasetFromDirectory(
			# 	new_path=new_directory, new_filename=new_filename, networktype=networktype, labels=labels, split_ratio=0.2,
			# 	debug_paths = debug_paths, path_benign_train=path_benign_train, path_malignant_train=path_malignant_train,
			# 	path_benign_val=path_benign_val, path_malignant_val=path_malignant_val,
			# 	path_benign_test=path_benign_test, path_malignant_test=path_malignant_test)

		if datasettype.value == DatasetType.MEDNODE.value:
			dbpath = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './complete_mednode_dataset')

			num_imgs = len(list(dbpath.glob('*/*.*'))) # counts all Kaggle Malignant Benign training images

			# train: 70 melanoma, 100 naevus
			assert num_imgs == 70+100

			logger.debug('%s %s', f"Images available in {datasettype.name} dataset:", num_imgs)

			# imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(dbpath, 't*/*/*.*'))}
			paths = glob(os.path.join(dbpath, '*/*.*'))
			# labels_dict = {os.path.basename(x): x for x in os.path.abspath(os.path.join(os.path.join(imageid_path_dict.values()), os.pardir))}
			df = pd.DataFrame()


			# Kaggle MB: Creating New Columns for better readability
			df['path'] = paths
			df['label'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir))))
			# df['portion'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir, os.pardir))))
			# assert df['label'].unique().shape[0] == 2
			df['cell_type_binary'] = np.where(df['label'] == 'melanoma', 'Melanoma', 'Non-Melanoma')
			df['cell_type_binary_idx'] = pd.CategoricalIndex(df.cell_type_binary, categories=self.classes_melanoma_binary).codes


			logger.debug("Check null data in Kaggle MB training metadata")
			display(df.isnull().sum())
			
			df['image'] = df.path.map(
				lambda x:(
					# img := Image.open(x).resize((img_width, img_height)).convert("RGB"), # [0]: PIL object
					img := load_img(path=x, target_size=(img_width, img_height)), # [0]: PIL object
					# np.asarray(img), # [1]: pixel array
					img_to_array(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df['img_sizes'] = df.path.map(
				lambda x:(
					Image.open(x).size
				)
			)

			labels = df.cell_type_binary.unique()

			if not isWholeRGBExist or not isTrainRGBExist or not isValRGBExist or not isTestRGBExist:
				for i in labels:
					os.makedirs(f"{whole_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{val_rgb_folder}/{i}", exist_ok=True)
					os.makedirs(f"{test_rgb_folder}/{i}", exist_ok=True)
					
			if not isWholeFeatureExist or not isTrainFeatureExist or not isValFeatureExist or not isTestFeatureExist:
				for i in labels:
					os.makedirs(f"{whole_feature_folder}/{i}", exist_ok=True)
					os.makedirs(f"{train_feature_folder}/{i}", exist_ok=True)

			
			# Dividing MEDNODE into train/val set
			df_trainset, df_validationset = train_test_split(df, test_size=0.2,random_state = 1)
			

			preprocessor.saveNumpyImagesToFiles(df_trainset, df, train_rgb_folder)
			preprocessor.saveNumpyImagesToFiles(df_validationset, df, val_rgb_folder)
			# preprocessor.saveNumpyImagesToFiles(df_testset, df, test_rgb_folder)

			# KaggleMB binary images/labels
			trainpixels = list(map(lambda x:x[1], df_trainset.image)) # Filter out only pixel from the list
			validationpixels = list(map(lambda x:x[1], df_validationset.image)) # Filter out only pixel from the list
			# testpixels = list(map(lambda x:x[1], df_testset.image)) # Filter out only pixel from the list
			
			trainimages = preprocessor.normalizeImgs(trainpixels, networktype)
			validationimages = preprocessor.normalizeImgs(validationpixels, networktype)
			testimages = None
			# testimages = preprocessor.normalizeImgs(testpixels, networktype)
			
			
			# trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
			# validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
			trainlabels_binary = to_categorical(df_trainset.cell_type_binary_idx, num_classes=2)
			validationlabels_binary = to_categorical(df_validationset.cell_type_binary_idx, num_classes=2)
			testlabels_binary = None
			# testlabels_binary = to_categorical(df_testset.cell_type_binary_idx, num_classes=2)

			assert len(trainpixels)+len(validationpixels) == 70+100
			assert len(trainpixels) == trainlabels_binary.shape[0]
			assert len(validationpixels) == validationlabels_binary.shape[0]
			assert trainimages.shape[0] == trainlabels_binary.shape[0]
			assert validationimages.shape[0] == validationlabels_binary.shape[0]

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert datasettype.name == 'MEDNODE'
			filename = path+'/'+f'{datasettype.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainimages, testimages, validationimages,
				trainlabels_binary, testlabels_binary, validationlabels_binary,
				2), file_bin)
			file_bin.close()

			if augment_ratio is not None and augment_ratio >= 1.0:
				
				augmented_db_name, df_mel_augmented, df_non_mel_augmented, trainimages_augmented, trainlabels_binary_augmented = \
					preprocessor.augmentation(datasettype, networktype, train_rgb_folder, labels, trainimages, trainlabels_binary, \
						augment_ratio, df_trainset)
				
				assert augmented_db_name.name == 'MEDNODE'
				filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainimages_augmented, testimages, validationimages,
					trainlabels_binary_augmented, testlabels_binary, validationlabels_binary,
					2), file_bin)
				file_bin.close()

			


		

		



		
	def normalizeImagesWithCustomMeanStd(self, trainingimages, validationimages, testimages, means, stds):
		# images is a list of images
		trainingimages = np.asarray(trainingimages).astype(np.float64)
		validationimages = np.asarray(validationimages).astype(np.float64)
		testimages = np.asarray(testimages).astype(np.float64)
		trainingimages = trainingimages[:, :, :, ::-1] # RGB to BGR
		validationimages = validationimages[:, :, :, ::-1] # RGB to BGR
		testimages = testimages[:, :, :, ::-1] # RGB to BGR

		meanB = means[0]
		meanG = means[1]
		meanR = means[2]
		stdB = stds[0]
		stdG = stds[1]
		stdR = stds[2]

		trainingimages[:, :, :, 0] -= meanB
		trainingimages[:, :, :, 1] -= meanG
		trainingimages[:, :, :, 2] -= meanR
		trainingimages[:, :, :, 0] /= stdB
		trainingimages[:, :, :, 1] /= stdG
		trainingimages[:, :, :, 2] /= stdR

		validationimages[:, :, :, 0] -= meanB
		validationimages[:, :, :, 1] -= meanG
		validationimages[:, :, :, 2] -= meanR
		validationimages[:, :, :, 0] /= stdB
		validationimages[:, :, :, 1] /= stdG
		validationimages[:, :, :, 2] /= stdR

		testimages[:, :, :, 0] -= meanB
		testimages[:, :, :, 1] -= meanG
		testimages[:, :, :, 2] -= meanR
		testimages[:, :, :, 0] /= stdB
		testimages[:, :, :, 1] /= stdG
		testimages[:, :, :, 2] /= stdR

		return trainingimages, validationimages, testimages
		

	def loadDatasetFromFile(self, filePath):
		trainimages, testimages, validationimages, \
			trainlabels, testlabels, validationlabels, num_classes = pickle.load(open(filePath, 'rb'))
		return trainimages, testimages, validationimages, \
			trainlabels, testlabels, validationlabels, num_classes

	def loadDatasetFromDirectory(self, networktype, split_ratio=None, debug_path=None, path_benign=None, path_malignant=None):
		#Transfer 'jpg' images to an array IMG
		# def Dataset_loader(imgPath):
		# 	img_width = self.image_size[1]
		# 	img_height = self.image_size[0]
		# 	# IMG = []
		# 	read = lambda imname: np.asarray(Image.open(imname).resize((img_width, img_height)).convert("RGB"))
		# 	for idx, IMAGE_NAME in enumerate(tqdm(os.listdir(imgPath))):
		# 		PATH = os.path.join(imgPath,IMAGE_NAME)
		# 		_, ftype = os.path.splitext(PATH)
		# 		if ftype == ".jpg":
		# 			img = read(PATH)
		# 			img = np.expand_dims(img, axis=0)
		# 			img = preprocess_input_resnet50(img)
		# 			if idx == 0:
		# 				IMG = img
		# 			elif idx > 0:
		# 				IMG = np.vstack((IMG, img))
		# 			# opencvImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		# 			# opencvImage = cv2.resize(opencvImage, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
		# 			# img = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2RGB)
		# 			# IMG.append(np.array(img)/255.)
		# 	return IMG

		x = None
		y = None
		x_portion = None
		y_portion = None

		if path_benign is not None or path_malignant is not None:
			
			preprocessor = Preprocess(self.image_size)
			

			# Preprocess images based on a network type


			benign_img = np.array(preprocessor.Dataset_loader(path_benign, networktype, debug_path))
			malignant_img = np.array(preprocessor.Dataset_loader(path_malignant, networktype, debug_path))
			# Create labels
			benign_label = np.zeros(len(benign_img))
			malignant_label = np.ones(len(malignant_img))
			benign_label_onehot = to_categorical(benign_label, num_classes=2)
			malignant_label_onehot = to_categorical(malignant_label, num_classes=2)
			# Concatenate imgs and labels
			X = np.concatenate((benign_img, malignant_img), axis = 0) # image
			Y = np.concatenate((benign_label_onehot, malignant_label_onehot), axis = 0) # label
			

			# Shuffle data
			s = np.arange(X.shape[0])
			np.random.shuffle(s)
			X_shuffled = X[s]
			Y_shuffled = Y[s]

			# Split validation data from train data
			if split_ratio is None:
				x = X_shuffled
				y = Y_shuffled
				x_portion = None
				y_portion = None
			elif split_ratio >= 0.0 and split_ratio <= 1.0:
				x, x_portion, y, y_portion = train_test_split(X_shuffled,Y_shuffled,test_size=split_ratio,random_state=10)
			elif split_ratio > 900:
				x=X_shuffled[split_ratio:] # X_shuffled[split_ratio] ~ X_shuffled[end]
				y=Y_shuffled[split_ratio:]
				x_portion=X_shuffled[:split_ratio] # x_shuffled[0] ~ x_shuffled[split_ratio-1]
				y_portion=Y_shuffled[:split_ratio]
			else:
				raise ValueError('split_ratio incorrect')
		

		return x, y, x_portion, y_portion

	def combineSavedDatasets(self, new_path, new_filename, *args):
		totalpath = new_path + new_filename

		trainimgs_list = []
		valimgs_list = []
		trainlabels_list = []
		vallabels_list = []
		# for db in args:
		# 	trainimages, _, validationimages, \
		# 	trainlabels, _, validationlabels, num_classes = pickle.load(open(db, 'rb'))
		# 	trainimgs_list.append(trainimages)
		# 	# testimgs_list.append(testimages)
		# 	valimgs_list.append(validationimages)
		# 	trainlabels_list.append(trainlabels)
		# 	# testlabels_list.append(testlabels)
		# 	vallabels_list.append(validationlabels)

		print('Combining...')
		for idx, pkl_file in enumerate(args):
			
			datum = pickle.load(open(pkl_file, 'rb'))
			print(f'Combining {idx+1} db out of {len(args)} dbs')
			# [0]:Trainimgs, [1]:Valimgs, [2]:testimgs, [3]:trainlabels, [4]:vallabels, [5]:testlabels
			trainimgs_list.append(datum[0])
			if datum[1] is not None:
				valimgs_list.append(datum[1])
			trainlabels_list.append(datum[3])
			if datum[4] is not None:
				vallabels_list.append(datum[4])
			

		
		
		assert sum(list(map(lambda v: len(v), trainimgs_list))) == sum(list(map(lambda v: len(v), trainlabels_list)))
		for idx, file in enumerate(trainimgs_list):
			assert trainimgs_list[idx].shape[0] == trainlabels_list[idx].shape[0]
		

		print('Stacking training images')
		list_size = sum(list(map(lambda v: len(v), trainimgs_list)))
		trainimgs_list = np.vstack(trainimgs_list)
		assert trainimgs_list.shape[0] == list_size
		# assert trainimages_combined.shape[0] == sum(list(map(lambda v: len(v), trainimgs_list)))
		# del trainimgs_list

		print('Stacking training labels')
		list_size = sum(list(map(lambda v: len(v), trainlabels_list)))
		trainlabels_list = np.vstack(trainlabels_list)
		assert trainlabels_list.shape[0] == list_size
		# assert trainlabels_combined.shape[0] == sum(list(map(lambda v: len(v), trainlabels_list)))
		# del trainlabels_list
		
		

		assert sum(list(map(lambda v: len(v), valimgs_list))) == sum(list(map(lambda v: len(v), vallabels_list)))
		for idx, file in enumerate(valimgs_list):
			assert valimgs_list[idx].shape[0] == vallabels_list[idx].shape[0]
		
		

		print('Stacking validation images')
		list_size = sum(list(map(lambda v: len(v), valimgs_list)))
		valimgs_list = np.vstack(valimgs_list)		
		assert valimgs_list.shape[0] == list_size
		# assert validationimages_combined.shape[0] == sum(list(map(lambda v: len(v), valimgs_list)))
		# del valimgs_list

		print('Stacking validation labels')
		list_size = sum(list(map(lambda v: len(v), vallabels_list)))
		vallabels_list = np.vstack(vallabels_list)
		assert vallabels_list.shape[0] == list_size
		# assert validationlabels_combined.shape[0] == sum(list(map(lambda v: len(v), vallabels_list)))
		# del vallabels_list

		

		leng = lambda x: len(x[x.len()])

		print(f'Pickling {new_filename}...')
		with open(totalpath, 'wb') as file:
				
				# pickle.dump((trainimages_combined, None, validationimages_combined,
				# trainlabels_combined, None, validationlabels_combined, 2), file)
				pickle.dump((trainimgs_list, None, valimgs_list,
				trainlabels_list, None, vallabels_list, 2), file)
		file.close()
		print(f'{new_filename} generated')
		
	def saveDatasetFromDirectory(self, new_path, new_filename, networktype, labels, split_ratio=None,
	debug_paths=None, path_benign_train=None, path_malignant_train=None,
	path_benign_val=None, path_malignant_val=None,
	path_benign_test=None, path_malignant_test=None):
		totalpath = new_path +'/' + new_filename

		# preprocessor = Preprocess(self.image_size)

		train_rgb_folder = debug_paths['train_rgb_folder']
		val_rgb_folder = debug_paths['val_rgb_folder']
		test_rgb_folder = debug_paths['test_rgb_folder']

		# split_ratio = None -> x_val = None, y_val = None
		x_train, y_train, x_val, y_val = self.loadDatasetFromDirectory(networktype, split_ratio, train_rgb_folder, path_benign_train, path_malignant_train)
		X_test, Y_test, _, _ = self.loadDatasetFromDirectory(networktype, None, test_rgb_folder, path_benign_test, path_malignant_test)
		
		# preprocessor.saveCustomDBImagesToFiles(labels, self.foldersExist, self.RGBfolders, trains, vals, tests)
		
		with open(totalpath, 'wb') as file:
				
				pickle.dump((x_train, X_test, x_val, y_train, Y_test, y_val, 2), file)
		file.close()

		# if augment_ratio is not None and augment_ratio >= 1.0:
				
		# 	_, df_mel_augmented, df_non_mel_augmented, trainimages_PAD_UFES_20_augmented, trainlabels_binary_PAD_UFES_20_augmented = \
		# 		preprocessor.augmentation(datasettype=None, networktype, train_rgb_folder, labels, x_train, y_train, \
		# 			augment_ratio, trainset_PAD_UFES_20)
			
			
		# 	filename_bin = path+'/'+f'{datasettype.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			
		# 	with open(filename_bin, 'wb') as file_bin:
				
		# 		pickle.dump((trainimages_PAD_UFES_20_augmented, None, validationimages_PAD_UFES_20,
		# 		trainlabels_binary_PAD_UFES_20_augmented, None, validationlabels_binary_PAD_UFES_20,
		# 		2), file_bin)
		# 	file_bin.close()


	


	def loadCSV(self, mode):
		if mode == DatasetType.HAM10000:
			# skin_df = pd.read_csv('hmnist_64_64_RGB.csv')
			skin_df = pd.read_csv(f'hmnist_{self.image_size[0]}_{self.image_size[1]}_RGB.csv')
			display(skin_df.head())
			
			X = skin_df.drop("label", axis=1).values
			label = skin_df["label"].values

			print("((# of images, w*h*c), (# of labels,))")
			print(X.shape, label.shape)

			# Scaling and Split Data into Train, Validation and Test set
			X_mean = np.mean(X)
			X_std = np.std(X)

			X = (X - X_mean)/X_std

			X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, label, test_size=0.1,random_state=0)
			print("In the case of test_size=0.1 -> 90% training imgs, 10% test imgs, 90% training labels, 10% test labels")
			print(X_train_orig.shape, X_test.shape, y_train_orig.shape, y_test.shape)

			X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.2, random_state=1)
			print("Split training images into 80:20 train/val sets")
			print("# of training images, # of validation images, # of training labels, # of validation labels")
			print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

			# Reshape the Data to Input in CNN
			X_train = X_train.reshape(X_train.shape[0], *(self.image_size[0], self.image_size[1], 3))
			X_val = X_val.reshape(X_val.shape[0], *(self.image_size[0], self.image_size[1], 3))
			X_test = X_test.reshape(X_test.shape[0], *(self.image_size[0], self.image_size[1], 3))
			
			print("Reshape images into # images, h, w, ch")
			print(X_train.shape, X_val.shape, X_test.shape)

			print("Again, reshape labels too")
			y_train = to_categorical(y_train)
			y_val = to_categorical(y_val)
			y_test = to_categorical(y_test)

			print("Reshaped labels shape")
			print(y_train.shape, y_val.shape, y_test.shape)

			data_gen_X_train = ImageDataGenerator(
				rotation_range = 90,    # randomly rotate images in the range (degrees, 0 to 180)
				zoom_range = 0.1,            # Randomly zoom image 
				width_shift_range = 0.1,   # randomly shift images horizontally
				height_shift_range = 0.1,  # randomly shift images vertically
				horizontal_flip= False,              # randomly flip images
				vertical_flip= False                 # randomly flip images
			)
			data_gen_X_train.fit(X_train)

			data_gen_X_val = ImageDataGenerator()
			data_gen_X_val.fit(X_val)

			return data_gen_X_train, data_gen_X_val, X_train, X_test, X_val, y_train, y_test, y_val, y_train.shape[1]
	
	def getImgSize(self):
		img_height = self.image_size[0]
		img_width = self.image_size[1]
		return img_height, img_width
		


	def loadTestData(self):
		# test_data_dir = pathlib.Path(path)
		num_test_img = len(list(self.test_data_dir.glob('*/*.jpg'))) # counts all images inside 'Test' folder
		print("Images available in test dataset:", num_test_img)



	

	

