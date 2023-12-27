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

import random
import math

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt


from IPython.core.debugger import Pdb
from IPython.display import display

import logging

from melanoma import augmentationStrategy as aug

class DatasetType(Enum):
	HAM10000 = 1
	ISIC2016= 2
	ISIC2017=3
	HAM10000_ISIC2016 = 100
	ALL = 9999

class ClassType(Enum):
	multi = 1
	binary = 2

class Util:
	# trainDataPath = ''
	# testDataPath = ''
	# train_ds = ''
	# val_ds = ''
	# class_names = []
	def __init__(self, path, image_size=(None, None), seed_val=1, split_portion=0.2, batch_size=32, color_mode='rgb'):
		self.base_dir = pathlib.Path(path)
		
		# self.trainDataPath = pathlib.Path.joinpath(path, '/Train')
		# self.testDataPath = pathlib.Path.joinpath(path, '/Test')
		self.seed_val = seed_val
		self.split_portion = split_portion
		self.image_size = image_size # (height, width)
		self.batch_size = batch_size
		self.color_mode = color_mode
		# self.epochs = epochs
		self.class_names = []
		# self.class_names = class_names
		self.train_ds = ''
		self.val_ds = ''

		#Lesion Dictionary created for ease
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
	
	def saveDatasetsToFile(self, mode, augment_ratio):
		# create logger
		logger = logging.getLogger('Melanoma classification')
		logger.setLevel(logging.DEBUG)

		def assert_(cond): assert cond
		def ExtractPixel(image):
					return [item[1] for item in image]

		path = str(self.base_dir) + '/melanomaDB' + '/customDB'
		# data_gen_HAM10000, HAM10000_multiclass, HAM10000_binaryclass, data_gen_ISIC2016, ISIC2016_binaryclass = self.load(mode)
		isExist = os.path.exists(path)
		if not isExist :
			os.makedirs(path)
		else:
			pass

		print("path: ", self.base_dir)
		print("seed value: ", self.seed_val)
		print("color_mode: ", self.color_mode)
		
		# Dataset path define
		from datetime import datetime
		now = datetime.now() # current date and time

		date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
		datasetname = mode.name

		debug_rgb_folder = path + f'/debug/{datasetname}/RGB/'+f'{self.image_size[0]}h_{self.image_size[1]}w_{date_time}'
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
		classes_melanoma_binary = ['Non-Melanoma', 'Melanoma']

		# Not required for pickled data
		# resize() order: (width, height)
		img_height = self.image_size[0]
		img_width = self.image_size[1]

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


		# height, width 순서
		image_shape = (img_height, img_width, 3)

		# HAM10000 multi-class images/labels
		if mode.value == DatasetType.HAM10000.value or mode.value == DatasetType.ALL.value:
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
			df_HAM10000['cell_type_binary_idx'] = pd.CategoricalIndex(df_HAM10000.cell_type_binary, categories=classes_melanoma_binary).codes
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


			

			df_HAM10000['ori_image'] = df_HAM10000.path.map(
				lambda x:(
				img := Image.open(x), # [0]: PIL object
				np.asarray(img), # [1]: pixel array
				)
			)
			
			df_HAM10000['image'] = df_HAM10000.path.map(
				lambda x:(
				img := Image.open(x).resize((img_width, img_height)), # [0]: PIL object
				np.asarray(img), # [1]: pixel array
				currentPath := pathlib.Path(x), # [2]: PosixPath
				# img.save(f"{whole_rgb_folder}/{currentPath.name}")
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
			trainset1_HAM10000, testset_HAM10000 = train_test_split(df_single_HAM10000, test_size=0.2,random_state = 80)
			trainset2_HAM10000, validationset_HAM10000 = train_test_split(trainset1_HAM10000, test_size=0.2,random_state = 600)
			trainset3_HAM10000 = df_HAM10000[df_HAM10000.num_images != 1]
			trainset_HAM10000 = pd.concat([trainset2_HAM10000, trainset3_HAM10000])

			trainset_HAM10000.index.map(lambda x: (
				currentPath_train := pathlib.Path(trainset_HAM10000.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
				label := trainset_HAM10000.cell_type_binary[x],
				assert_(label == df_HAM10000.cell_type_binary[x]),
				trainset_HAM10000.image[x][0].save(f"{train_rgb_folder}/{label}/{currentPath_train.name}", quality=100, subsampling=0)
			))

			validationset_HAM10000.index.map(lambda x: (
				currentPath_val := pathlib.Path(validationset_HAM10000.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
				label := validationset_HAM10000.cell_type_binary[x],
				assert_(label == df_HAM10000.cell_type_binary[x]),
				validationset_HAM10000.image[x][0].save(f"{val_rgb_folder}/{label}/{currentPath_val.name}", quality=100, subsampling=0)
			))

			testset_HAM10000.index.map(lambda x: (
				currentPath_test := pathlib.Path(testset_HAM10000.image[x][2]), # [0]: PIL obj, [1]: pixels, [2]: PosixPath
				label := testset_HAM10000.cell_type_binary[x],
				assert_(label == df_HAM10000.cell_type_binary[x]),
				testset_HAM10000.image[x][0].save(f"{test_rgb_folder}/{label}/{currentPath_test.name}", quality=100, subsampling=0)
			))

			trainpixels_HAM10000 = list(map(lambda x:x[1], trainset_HAM10000.image)) # Filter out only pixel from the list
			testpixels_HAM10000 = list(map(lambda x:x[1], testset_HAM10000.image))
			validationpixels_HAM10000 = list(map(lambda x:x[1], validationset_HAM10000.image))

			means, stds = getMeanStd(trainpixels_HAM10000)
			trainlabels_multi_HAM10000 = np.asarray(trainset_HAM10000.cell_type_idx)
			testlabels_multi_HAM10000 = np.asarray(testset_HAM10000.cell_type_idx)
			validationlabels_multi_HAM10000 = np.asarray(validationset_HAM10000.cell_type_idx)
			# HAM10000 binary labels (Don't need to generate images since they are identical regardless of multi/binary labels)
			trainlabels_binary_HAM10000 = np.asarray(trainset_HAM10000.cell_type_binary_idx)
			testlabels_binary_HAM10000 = np.asarray(testset_HAM10000.cell_type_binary_idx)
			validationlabels_binary_HAM10000 = np.asarray(validationset_HAM10000.cell_type_binary_idx)

			assert num_train_img_HAM10000 == (len(trainpixels_HAM10000) + len(testpixels_HAM10000) + len(validationpixels_HAM10000))
			assert len(trainpixels_HAM10000) == trainlabels_multi_HAM10000.shape[0]
			assert len(trainpixels_HAM10000) == trainlabels_binary_HAM10000.shape[0]
			assert len(validationpixels_HAM10000) == validationlabels_multi_HAM10000.shape[0]
			assert len(validationpixels_HAM10000) == validationlabels_binary_HAM10000.shape[0]
			assert len(testpixels_HAM10000) == testlabels_multi_HAM10000.shape[0]
			assert len(testpixels_HAM10000) == testlabels_binary_HAM10000.shape[0]

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

			filename_bin = path+'/'+f'{mode.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			filename_multi = path+'/'+f'{mode.name}_{self.image_size[0]}h_{self.image_size[1]}w_multiclass.pkl' # height x width
			with open(filename_bin, 'wb') as file_bin:
				
				pickle.dump((trainpixels_HAM10000, testpixels_HAM10000, validationpixels_HAM10000,
				trainlabels_binary_HAM10000, testlabels_binary_HAM10000, validationlabels_binary_HAM10000,
				means, stds, 2), file_bin)
			file_bin.close()

			with open(filename_multi, 'wb') as file_multi:
				
				pickle.dump((trainpixels_HAM10000, testpixels_HAM10000, validationpixels_HAM10000,
				trainlabels_multi_HAM10000, testlabels_multi_HAM10000, validationlabels_multi_HAM10000,
				means, stds, num_classes_multi_HAM10000), file_multi)
			file_multi.close()

			# Augmentation only on training set
			if augment_ratio is not None and augment_ratio >= 1.0:
				# mel_cnt = df_HAM10000['cell_type_binary'].value_counts()['Melanoma']
				# non_mel_cnt = df_HAM10000['cell_type_binary'].value_counts()['Non-Melanoma']
				mel_cnt = trainset_HAM10000[trainset_HAM10000.cell_type_binary=='Melanoma'].shape[0]
				non_mel_cnt = trainset_HAM10000[trainset_HAM10000.cell_type_binary=='Non-Melanoma'].shape[0]

				
				augMethod = aug.Augmentation(aug.crop_flip_brightnesscontrast())

				df_mel = trainset_HAM10000[trainset_HAM10000.cell_type_binary=='Melanoma']
				df_non_mel = trainset_HAM10000[trainset_HAM10000.cell_type_binary=='Non-Melanoma']

				df_mel_augmented = pd.DataFrame(columns=trainset_HAM10000.columns.tolist())
				df_non_mel_augmented = pd.DataFrame(columns=trainset_HAM10000.columns.tolist())

				trainset_HAM10000_cp = trainset_HAM10000.copy()
				
				trainset_HAM10000_cp['image'] = ExtractPixel(trainset_HAM10000['image'])

				if mel_cnt < non_mel_cnt:
					# melanoma augmentation here
					# Melanoma images will be augmented to N times of the Non-melanoma images
					for j, id in enumerate(range((non_mel_cnt - mel_cnt), math.ceil(non_mel_cnt * augment_ratio))):
						randmel_idx = random.choice(df_mel.index)
						df_mel_augmented.loc[j] = trainset_HAM10000.loc[randmel_idx, ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset', 'num_images', 'path', 'cell_type', 'cell_type_binary', 'cell_type_idx', 'cell_type_binary_idx', 'image']]
						augmented_img = augMethod.augmentation(input_img=trainset_HAM10000['ori_image'][randmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						# df_mel_augmented = pd.concat([df_mel_augmented, augmented_img])
						df_mel_augmented.at[j, 'image'] = None
						df_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - (non_mel_cnt - mel_cnt)
						assert df_mel_augmented.shape[0] <= num_augmented_img
					# non-melanoma augmentation here
					for j, id in enumerate(range(non_mel_cnt, math.ceil(non_mel_cnt * augment_ratio))):
						randnonmel_idx = random.choice(df_non_mel.index)
						df_non_mel_augmented.loc[j] = trainset_HAM10000.loc[randnonmel_idx, ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset', 'num_images', 'path', 'cell_type', 'cell_type_binary', 'cell_type_idx', 'cell_type_binary_idx', 'image']]
						augmented_img = augMethod.augmentation(input_img=trainset_HAM10000['ori_image'][randnonmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_non_mel_augmented.at[j, 'image'] = None
						df_non_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - non_mel_cnt
						assert df_non_mel_augmented.shape[0] <= num_augmented_img
				elif mel_cnt > non_mel_cnt:
					# melanoma augmentation here
					for j, id in enumerate(range(mel_cnt, math.ceil(mel_cnt * augment_ratio))):
						randmel_idx = random.choice(df_mel.index)
						df_mel_augmented.loc[j] = trainset_HAM10000.loc[randmel_idx, ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset', 'num_images', 'path', 'cell_type', 'cell_type_binary', 'cell_type_idx', 'cell_type_binary_idx', 'image']]
						augmented_img = augMethod.augmentation(input_img=trainset_HAM10000['ori_image'][randmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_mel_augmented.at[j, 'image'] = None
						df_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(mel_cnt * augment_ratio) - mel_cnt
						assert df_mel_augmented.shape[0] <= num_augmented_img
					# non-melanoma augmentation here
					for j, i in enumerate(range((mel_cnt - non_mel_cnt), math.ceil(mel_cnt * augment_ratio))):
						randnonmel_idx = random.choice(df_non_mel.index)
						df_non_mel_augmented.loc[j] = trainset_HAM10000.loc[randnonmel_idx, ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset', 'num_images', 'path', 'cell_type', 'cell_type_binary', 'cell_type_idx', 'cell_type_binary_idx', 'image']]
						augmented_img = augMethod.augmentation(input_img=trainset_HAM10000['ori_image'][randnonmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_non_mel_augmented.at[j, 'image'] = None
						df_non_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(mel_cnt * augment_ratio) - (mel_cnt - non_mel_cnt)
						assert df_non_mel_augmented.shape[0] <= num_augmented_img

				trainset_HAM10000_augmented = pd.concat([trainset_HAM10000_cp, df_mel_augmented, df_non_mel_augmented])

				augmentation_folder = f"{train_rgb_folder}/augmented"
				isAugFolderExist = os.path.exists(augmentation_folder)
				if not isAugFolderExist:
					for i in labels:
						os.makedirs(f"{augmentation_folder}/{i}", exist_ok=True)

				# Save augmented images for viewing purpose
				for idx in df_mel_augmented.index:
					img = Image.fromarray(df_mel_augmented.image[idx], mode='RGB')
					currentPath = pathlib.Path(df_mel_augmented.path[idx])
					label = df_mel_augmented.cell_type_binary[idx]
					assert label == 'Melanoma'
					img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)

				for idx in df_non_mel_augmented.index:
					img = Image.fromarray(df_non_mel_augmented.image[idx], mode='RGB')
					currentPath = pathlib.Path(df_non_mel_augmented.path[idx])
					label = df_non_mel_augmented.cell_type_binary[idx]
					assert label == 'Non-Melanoma'
					img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)
			
				trainpixels_HAM10000_augmented = list(map(lambda x:x, trainset_HAM10000_augmented.image)) # Filter out only pixel from the list

				new_means, new_stds = getMeanStd(trainpixels_HAM10000_augmented)
				
				trainlabels_binary_HAM10000_augmented = np.asarray(trainset_HAM10000_augmented.cell_type_binary_idx, dtype='int8')

				assert trainset_HAM10000_augmented.shape[0] == trainlabels_binary_HAM10000_augmented.shape[0]
				assert len(trainpixels_HAM10000_augmented) == trainlabels_binary_HAM10000_augmented.shape[0]
			
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
				# trainimages_HAM10000_augmented = trainimages_HAM10000_augmented.reshape(trainimages_HAM10000_augmented.shape[0], *image_shape)

				filename_bin = path+'/'+f'{mode.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainpixels_HAM10000_augmented, testpixels_HAM10000, validationpixels_HAM10000,
					trainlabels_binary_HAM10000_augmented, testlabels_binary_HAM10000, validationlabels_binary_HAM10000,
					new_means, new_stds, 2), file_bin)
				file_bin.close()

			



		
		if mode.value == DatasetType.ISIC2016.value or mode.value == DatasetType.ALL.value:
			ISIC2016_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Training_Data')
			ISIC2016_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Test_Data')

			num_train_img_ISIC2016 = len(list(ISIC2016_training_path.glob('./*.jpg'))) # counts all ISIC2016 training images
			num_test_img_ISIC2016 = len(list(ISIC2016_test_path.glob('./*.jpg'))) # counts all ISIC2016 test images

			logger.debug('%s %s', "Images available in ISIC2016 train dataset:", num_train_img_ISIC2016)
			logger.debug('%s %s', "Images available in ISIC2016 test dataset:", num_test_img_ISIC2016)

			# ISIC2016: Dictionary for Image Names
			imageid_path_training_dict_ISIC2016 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2016_training_path, '*.jpg'))}
			imageid_path_test_dict_ISIC2016 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2016_test_path, '*.jpg'))}

			df_training_ISIC2016 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Training_GroundTruth.csv')))
			df_test_ISIC2016 = pd.read_csv(str(pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Test_GroundTruth.csv')))

			logger.debug("Let's check ISIC2016 metadata briefly")
			logger.debug("This is ISIC2016 training data")
			# Creating default column titles
			df_training_ISIC2016.columns = ['image_id', 'label']
			display(df_training_ISIC2016.head())
			df_test_ISIC2016.columns = ['image_id', 'label']
			logger.debug("This is ISIC2016 test data")
			display(df_test_ISIC2016.head())


			classes_binary_ISIC2016 = df_training_ISIC2016.label.unique() # second column is label
			num_classes_binary_ISIC2016 = len(classes_binary_ISIC2016)
			classes_binary_ISIC2016, num_classes_binary_ISIC2016

			# ISIC2016: Creating New Columns for better readability
			df_training_ISIC2016['path'] = df_training_ISIC2016.image_id.map(imageid_path_training_dict_ISIC2016.get)
			df_training_ISIC2016['cell_type_binary'] = df_training_ISIC2016.label.map(self.lesion_type_binary_dict_training_ISIC2016.get)
			# Define codes for compatibility among datasets
			df_training_ISIC2016['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2016.cell_type_binary, categories=classes_melanoma_binary).codes
			df_test_ISIC2016['path'] = df_test_ISIC2016.image_id.map(imageid_path_test_dict_ISIC2016.get)
			df_test_ISIC2016['cell_type_binary'] = df_test_ISIC2016.label.map(self.lesion_type_binary_dict_test_ISIC2016.get)
			# Define codes for compatibility among datasets
			df_test_ISIC2016['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2016.cell_type_binary, categories=classes_melanoma_binary).codes
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

			df_training_ISIC2016['ori_image'] = df_training_ISIC2016.path.map(
				lambda x:(
					img := Image.open(x), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
				)
			)
			
			df_training_ISIC2016['image'] = df_training_ISIC2016.path.map(
				lambda x:(
					img := Image.open(x).resize((img_width, img_height)), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
				)
			)

			df_test_ISIC2016['ori_image'] = df_test_ISIC2016.path.map(
				lambda x:(
					img := Image.open(x), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
				)
			)
			
			df_test_ISIC2016['image'] = df_test_ISIC2016.path.map(
				lambda x:(
					img := Image.open(x).resize((img_width, img_height)), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
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
			trainset_ISIC2016, validationset_ISIC2016 = train_test_split(df_training_ISIC2016, test_size=0.2,random_state = 80)
			# ISIC2016 test data is given, so there is no need to create test dataset separately
			testset_ISIC2016 = df_test_ISIC2016





			# ISIC2016 binary images/labels
			trainpixels_ISIC2016 = list(map(lambda x:x[1], trainset_ISIC2016.image)) # Filter out only pixel from the list
			testpixels_ISIC2016 = list(map(lambda x:x[1], testset_ISIC2016.image))
			validationpixels_ISIC2016 = list(map(lambda x:x[1], validationset_ISIC2016.image))

			means, stds = getMeanStd(trainpixels_ISIC2016)
			trainlabels_binary_ISIC2016 = np.asarray(trainset_ISIC2016.cell_type_binary_idx)
			testlabels_binary_ISIC2016 = np.asarray(testset_ISIC2016.cell_type_binary_idx)
			validationlabels_binary_ISIC2016 = np.asarray(validationset_ISIC2016.cell_type_binary_idx)

			assert len(trainpixels_ISIC2016) == trainlabels_binary_ISIC2016.shape[0]
			assert len(validationpixels_ISIC2016) == validationlabels_binary_ISIC2016.shape[0]
			assert len(testpixels_ISIC2016) == testlabels_binary_ISIC2016.shape[0]

			# trainimages_ISIC2016 = trainimages_ISIC2016.reshape(trainimages_ISIC2016.shape[0], *image_shape)

			filename = path+'/'+f'{mode.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainpixels_ISIC2016, testpixels_ISIC2016, validationpixels_ISIC2016,
				trainlabels_binary_ISIC2016, testlabels_binary_ISIC2016,validationlabels_binary_ISIC2016,
				means, stds, 2), file_bin)
			file_bin.close()


			# Augmentation only on training set
			if augment_ratio is not None and augment_ratio >= 1.0:
				mel_cnt = trainset_ISIC2016[trainset_ISIC2016.cell_type_binary=='Melanoma'].shape[0]
				non_mel_cnt = trainset_ISIC2016[trainset_ISIC2016.cell_type_binary=='Non-Melanoma'].shape[0]

				
				augMethod = aug.Augmentation(aug.crop_flip_brightnesscontrast())

				df_mel = trainset_ISIC2016[trainset_ISIC2016.cell_type_binary=='Melanoma']
				df_non_mel = trainset_ISIC2016[trainset_ISIC2016.cell_type_binary=='Non-Melanoma']

				df_mel_augmented = pd.DataFrame(columns=trainset_ISIC2016.columns.tolist())
				df_non_mel_augmented = pd.DataFrame(columns=trainset_ISIC2016.columns.tolist())

				trainset_ISIC2016_cp = trainset_ISIC2016.copy()
				
				
				trainset_ISIC2016_cp['image'] = ExtractPixel(trainset_ISIC2016['image'])

				if mel_cnt < non_mel_cnt:
					# melanoma augmentation here
					# Melanoma images will be augmented to N times of the Non-melanoma images
					for j, id in enumerate(range((non_mel_cnt - mel_cnt), math.ceil(non_mel_cnt * augment_ratio))):
						randmel_idx = random.choice(df_mel.index)
						df_mel_augmented.loc[j] = trainset_ISIC2016.loc[randmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2016['ori_image'][randmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						# df_mel_augmented = pd.concat([df_mel_augmented, augmented_img])
						df_mel_augmented.at[j, 'image'] = None
						df_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - (non_mel_cnt - mel_cnt)
						assert df_mel_augmented.shape[0] <= num_augmented_img
					# non-melanoma augmentation here
					for j, id in enumerate(range(non_mel_cnt, math.ceil(non_mel_cnt * augment_ratio))):
						randnonmel_idx = random.choice(df_non_mel.index)
						df_non_mel_augmented.loc[j] = trainset_ISIC2016.loc[randnonmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2016['ori_image'][randnonmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_non_mel_augmented.at[j, 'image'] = None
						df_non_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - non_mel_cnt
						assert df_non_mel_augmented.shape[0] <= num_augmented_img
				elif mel_cnt > non_mel_cnt:
					# melanoma augmentation here
					for j, id in enumerate(range(mel_cnt, math.ceil(mel_cnt * augment_ratio))):
						randmel_idx = random.choice(df_mel.index)
						df_mel_augmented.loc[j] = trainset_ISIC2016.loc[randmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2016['ori_image'][randmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_mel_augmented.at[j, 'image'] = None
						df_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(mel_cnt * augment_ratio) - mel_cnt
						assert df_mel_augmented.shape[0] <= num_augmented_img
					# non-melanoma augmentation here
					for j, i in enumerate(range((mel_cnt - non_mel_cnt), math.ceil(mel_cnt * augment_ratio))):
						randnonmel_idx = random.choice(df_non_mel.index)
						df_non_mel_augmented.loc[j] = trainset_ISIC2016.loc[randnonmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2016['ori_image'][randnonmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_non_mel_augmented.at[j, 'image'] = None
						df_non_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(mel_cnt * augment_ratio) - (mel_cnt - non_mel_cnt)
						assert df_non_mel_augmented.shape[0] <= num_augmented_img

				trainset_ISIC2016_augmented = pd.concat([trainset_ISIC2016_cp, df_mel_augmented, df_non_mel_augmented])

				augmentation_folder = f"{train_rgb_folder}/augmented"
				isAugFolderExist = os.path.exists(augmentation_folder)
				if not isAugFolderExist:
					for i in labels:
						os.makedirs(f"{augmentation_folder}/{i}", exist_ok=True)

				# Save augmented images for viewing purpose
				for idx in df_mel_augmented.index:
					img = Image.fromarray(df_mel_augmented.image[idx], mode='RGB')
					currentPath = pathlib.Path(df_mel_augmented.path[idx])
					label = df_mel_augmented.cell_type_binary[idx]
					assert label == 'Melanoma'
					img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)

				for idx in df_non_mel_augmented.index:
					img = Image.fromarray(df_non_mel_augmented.image[idx], mode='RGB')
					currentPath = pathlib.Path(df_non_mel_augmented.path[idx])
					label = df_non_mel_augmented.cell_type_binary[idx]
					assert label == 'Non-Melanoma'
					img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)
			
				trainpixels_ISIC2016_augmented = list(map(lambda x:x, trainset_ISIC2016_augmented.image)) # Filter out only pixel from the list

				new_means, new_stds = getMeanStd(trainpixels_ISIC2016_augmented)
				
				trainlabels_binary_ISIC2016_augmented = np.asarray(trainset_ISIC2016_augmented.cell_type_binary_idx, dtype='int8')

				assert trainset_ISIC2016_augmented.shape[0] == trainlabels_binary_ISIC2016_augmented.shape[0]
				assert len(trainpixels_ISIC2016_augmented) == trainlabels_binary_ISIC2016_augmented.shape[0]
			
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
				# trainimages_HAM10000_augmented = trainimages_HAM10000_augmented.reshape(trainimages_HAM10000_augmented.shape[0], *image_shape)

				assert mode.name == 'ISIC2016'
				filename_bin = path+'/'+f'{mode.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainpixels_ISIC2016_augmented, testpixels_ISIC2016, validationpixels_ISIC2016,
					trainlabels_binary_ISIC2016_augmented, testlabels_binary_ISIC2016, validationlabels_binary_ISIC2016,
					new_means, new_stds, 2), file_bin)
				file_bin.close()


		if mode.value == DatasetType.ISIC2017.value or mode.value == DatasetType.ALL.value:
			ISIC2017_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Training_Data')
			ISIC2017_val_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Validation_Data')
			ISIC2017_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2017', './ISIC-2017_Test_v2_Data')

			num_train_img_ISIC2017 = len(list(ISIC2017_training_path.glob('./*.jpg'))) # counts all ISIC2017 training images
			num_val_img_ISIC2017 = len(list(ISIC2017_val_path.glob('./*.jpg'))) # counts all ISIC2017 validation images
			num_test_img_ISIC2017 = len(list(ISIC2017_test_path.glob('./*.jpg'))) # counts all ISIC2017 test images

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
			df_training_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_binary, categories=classes_melanoma_binary).codes
			df_training_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
			df_training_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_training_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes

			df_val_ISIC2017['path'] = df_val_ISIC2017.image_id.map(imageid_path_val_dict_ISIC2017.get)
			df_val_ISIC2017['cell_type_binary'] = df_val_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
			df_val_ISIC2017['cell_type_task3_1'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
			df_val_ISIC2017['cell_type_task3_2'] = df_val_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
			df_val_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_binary, categories=classes_melanoma_binary).codes
			df_val_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
			df_val_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_val_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes

			df_test_ISIC2017['path'] = df_test_ISIC2017.image_id.map(imageid_path_test_dict_ISIC2017.get)
			df_test_ISIC2017['cell_type_binary'] = df_test_ISIC2017.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
			df_test_ISIC2017['cell_type_task3_1'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
			df_test_ISIC2017['cell_type_task3_2'] = df_test_ISIC2017.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
			df_test_ISIC2017['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_binary, categories=classes_melanoma_binary).codes
			df_test_ISIC2017['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_1, categories=classes_ISIC2017_task3_1).codes
			df_test_ISIC2017['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_test_ISIC2017.cell_type_task3_2, categories=classes_ISIC2017_task3_2).codes



			logger.debug("Check null data in ISIC2017 training metadata")
			display(df_training_ISIC2017.isnull().sum())
			logger.debug("Check null data in ISIC2017 validation metadata")
			display(df_val_ISIC2017.isnull().sum())
			logger.debug("Check null data in ISIC2017 test metadata")
			display(df_test_ISIC2017.isnull().sum())


			df_training_ISIC2017['ori_image'] = df_training_ISIC2017.path.map(
				lambda x:(
					img := Image.open(x), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
				)
			)
			
			df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(
				lambda x:(
					img := Image.open(x).resize((img_width, img_height)), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
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
					img := Image.open(x).resize((img_width, img_height)), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
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
					img := Image.open(x).resize((img_width, img_height)), # [0]: PIL object
					np.asarray(img), # [1]: pixel array
					currentPath := pathlib.Path(x), # [2]: PosixPath
					# img.save(f"{whole_rgb_folder}/{currentPath.name}")
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



			# ISIC2017 binary images/labels
			trainpixels_ISIC2017 = list(map(lambda x:x[1], trainset_ISIC2017.image)) # Filter out only pixel from the list
			validationpixels_ISIC2017 = list(map(lambda x:x[1], validationset_ISIC2017.image)) # Filter out only pixel from the list
			testpixels_ISIC2017 = list(map(lambda x:x[1], testset_ISIC2017.image)) # Filter out only pixel from the list
			means, stds = getMeanStd(trainpixels_ISIC2017)
			# trainimages_ISIC2017 = normalizeImages(list(trainset_ISIC2017.image))
			# testimages_ISIC2017 = normalizeImages(list(testset_ISIC2017.image))
			# validationimages_ISIC2017 = normalizeImages(list(validationset_ISIC2017.image))
			trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx)
			testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx)
			validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx)

			# trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

			assert mode.name == 'ISIC2017'
			filename = path+'/'+f'{mode.name}_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
			with open(filename, 'wb') as file_bin:
				
				pickle.dump((trainpixels_ISIC2017, testpixels_ISIC2017, validationpixels_ISIC2017,
				trainlabels_binary_ISIC2017, testlabels_binary_ISIC2017, validationlabels_binary_ISIC2017,
				means, stds, 2), file_bin)
			file_bin.close()

			# Augmentation only on training set
			if augment_ratio is not None and augment_ratio >= 1.0:
				mel_cnt = trainset_ISIC2017[trainset_ISIC2017.cell_type_binary=='Melanoma'].shape[0]
				non_mel_cnt = trainset_ISIC2017[trainset_ISIC2017.cell_type_binary=='Non-Melanoma'].shape[0]

				
				augMethod = aug.Augmentation(aug.crop_flip_brightnesscontrast())

				df_mel = trainset_ISIC2017[trainset_ISIC2017.cell_type_binary=='Melanoma']
				df_non_mel = trainset_ISIC2017[trainset_ISIC2017.cell_type_binary=='Non-Melanoma']

				df_mel_augmented = pd.DataFrame(columns=trainset_ISIC2017.columns.tolist())
				df_non_mel_augmented = pd.DataFrame(columns=trainset_ISIC2017.columns.tolist())

				trainset_ISIC2017_cp = trainset_ISIC2017.copy()
				
				
				trainset_ISIC2017_cp['image'] = ExtractPixel(trainset_ISIC2017['image'])

				if mel_cnt < non_mel_cnt:
					# melanoma augmentation here
					# Melanoma images will be augmented to N times of the Non-melanoma images
					for j, id in enumerate(range((non_mel_cnt - mel_cnt), math.ceil(non_mel_cnt * augment_ratio))):
						randmel_idx = random.choice(df_mel.index)
						df_mel_augmented.loc[j] = trainset_ISIC2017.loc[randmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2017['ori_image'][randmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						# df_mel_augmented = pd.concat([df_mel_augmented, augmented_img])
						df_mel_augmented.at[j, 'image'] = None
						df_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - (non_mel_cnt - mel_cnt)
						assert df_mel_augmented.shape[0] <= num_augmented_img
					# non-melanoma augmentation here
					for j, id in enumerate(range(non_mel_cnt, math.ceil(non_mel_cnt * augment_ratio))):
						randnonmel_idx = random.choice(df_non_mel.index)
						df_non_mel_augmented.loc[j] = trainset_ISIC2017.loc[randnonmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2017['ori_image'][randnonmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_non_mel_augmented.at[j, 'image'] = None
						df_non_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(non_mel_cnt * augment_ratio) - non_mel_cnt
						assert df_non_mel_augmented.shape[0] <= num_augmented_img
				elif mel_cnt > non_mel_cnt:
					# melanoma augmentation here
					for j, id in enumerate(range(mel_cnt, math.ceil(mel_cnt * augment_ratio))):
						randmel_idx = random.choice(df_mel.index)
						df_mel_augmented.loc[j] = trainset_ISIC2017.loc[randmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2017['ori_image'][randmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_mel_augmented.at[j, 'image'] = None
						df_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(mel_cnt * augment_ratio) - mel_cnt
						assert df_mel_augmented.shape[0] <= num_augmented_img
					# non-melanoma augmentation here
					for j, i in enumerate(range((mel_cnt - non_mel_cnt), math.ceil(mel_cnt * augment_ratio))):
						randnonmel_idx = random.choice(df_non_mel.index)
						df_non_mel_augmented.loc[j] = trainset_ISIC2017.loc[randnonmel_idx]
						augmented_img = augMethod.augmentation(input_img=trainset_ISIC2017['ori_image'][randnonmel_idx][1], crop_height=img_height, crop_width=img_width, zoomout=1.0, zoomin=1.1, p_scaling=0.5, p_rotation=0.5, p_hflip=0.5, p_vflip=0.5,\
    															p_randomBrightnessContrast=0.2)
						df_non_mel_augmented.at[j, 'image'] = None
						df_non_mel_augmented.at[j, 'image'] = augmented_img['image']

						num_augmented_img = math.ceil(mel_cnt * augment_ratio) - (mel_cnt - non_mel_cnt)
						assert df_non_mel_augmented.shape[0] <= num_augmented_img

				trainset_ISIC2017_augmented = pd.concat([trainset_ISIC2017_cp, df_mel_augmented, df_non_mel_augmented])

				augmentation_folder = f"{train_rgb_folder}/augmented"
				isAugFolderExist = os.path.exists(augmentation_folder)
				if not isAugFolderExist:
					for i in labels:
						os.makedirs(f"{augmentation_folder}/{i}", exist_ok=True)

				# Save augmented images for viewing purpose
				for idx in df_mel_augmented.index:
					img = Image.fromarray(df_mel_augmented.image[idx], mode='RGB')
					currentPath = pathlib.Path(df_mel_augmented.path[idx])
					label = df_mel_augmented.cell_type_binary[idx]
					assert label == 'Melanoma'
					img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)

				for idx in df_non_mel_augmented.index:
					img = Image.fromarray(df_non_mel_augmented.image[idx], mode='RGB')
					currentPath = pathlib.Path(df_non_mel_augmented.path[idx])
					label = df_non_mel_augmented.cell_type_binary[idx]
					assert label == 'Non-Melanoma'
					img.save(f"{augmentation_folder}/{label}/{idx}_{currentPath.stem}.jpg", quality=100, subsampling=0)
			
				trainpixels_ISIC2017_augmented = list(map(lambda x:x, trainset_ISIC2017_augmented.image)) # Filter out only pixel from the list

				new_means, new_stds = getMeanStd(trainpixels_ISIC2017_augmented)
				
				trainlabels_binary_ISIC2017_augmented = np.asarray(trainset_ISIC2017_augmented.cell_type_binary_idx, dtype='int8')

				assert trainset_ISIC2017_augmented.shape[0] == trainlabels_binary_ISIC2017_augmented.shape[0]
				assert len(trainpixels_ISIC2017_augmented) == trainlabels_binary_ISIC2017_augmented.shape[0]
			
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
				# trainimages_HAM10000_augmented = trainimages_HAM10000_augmented.reshape(trainimages_HAM10000_augmented.shape[0], *image_shape)

				assert mode.name == 'ISIC2017'
				filename_bin = path+'/'+f'{mode.name}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.image_size[0]}h_{self.image_size[1]}w_binary.pkl' # height x width
				
				with open(filename_bin, 'wb') as file_bin:
					
					pickle.dump((trainpixels_ISIC2017_augmented, testpixels_ISIC2017, validationpixels_ISIC2017,
					trainlabels_binary_ISIC2017_augmented, testlabels_binary_ISIC2017, validationlabels_binary_ISIC2017,
					new_means, new_stds, 2), file_bin)
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
			trainlabels, testlabels, validationlabels, means, stds, num_classes = pickle.load(open(filePath, 'rb'))
		return trainimages, testimages, validationimages, \
			trainlabels, testlabels, validationlabels, means, stds, num_classes
	
	def combine_images(self, **kwargs):
		
		for idx, (key, value) in enumerate(kwargs.items()):
			print("Input " + str(idx) + ": " + key)
			if idx==0:
				name = key + " AND "
				combined = value
			elif idx>0:
				name = name + key + " AND "
				combined = np.vstack((combined, value))
			
		print("Combined images: " + name)
		return combined
	
	def combine_labels(self, **kwargs):
		for idx, (key, value) in enumerate(kwargs.items()):
			print("Input: " + str(idx) + ": " + key)
			if idx==0:
				name = key + " AND "
				combined = value
			elif idx>0:
				name = name + key + " AND "
				combined = np.hstack((combined, value))
			
		print("Combined labels: " + name)
		return combined

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



	

	

