from glob import glob
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from enum import Enum
import seaborn as sns

import PIL
from PIL import Image

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt


from IPython.core.debugger import Pdb
from IPython.display import display

import logging

class DatasetType(Enum):
	HAM10000 = 1
	ISIC2016= 2
	HAM10000_ISIC2016 = 100

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

		self.lesion_type_binary_dict_training_ISIC2016 = {
			'benign' : 'Non-Melanoma',
			'malignant' : 'Melanoma',
		}
		self.lesion_type_binary_dict_test_ISIC2016 = {
			0.0 : 'Non-Melanoma',
			1.0 : 'Melanoma',
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
	
	def load(self, mode):
		# create logger
		logger = logging.getLogger('Melanoma classification')
		logger.setLevel(logging.DEBUG)


		# if mode == DatasetType.HAM10000:
		# Set default image size for HAM10000
		# self.image_size = (112, 150) # height, width
		# logger.debug('%s %s', "path: ", self.base_dir)
		# logger.debug('%s %s', "seed value: ", self.seed_val)
		# logger.debug('%s %s', "color_mode: ", self.color_mode)
		print("path: ", self.base_dir)
		print("seed value: ", self.seed_val)
		print("color_mode: ", self.color_mode)
		
		# Dataset path define
		HAM10000_path = pathlib.Path.joinpath(pathlib.Path.cwd(), './HAM10000_images_combined')
		ISIC2016_training_path = pathlib.Path.joinpath(pathlib.Path.cwd(), './ISIC2016', './ISBI2016_ISIC_Part3_Training_Data')
		ISIC2016_test_path = pathlib.Path.joinpath(pathlib.Path.cwd(), './ISIC2016', './ISBI2016_ISIC_Part3_Test_Data')
		num_train_img_HAM10000 = len(list(HAM10000_path.glob('./*.jpg'))) # counts all HAM10000 images
		num_train_img_ISIC2016 = len(list(ISIC2016_training_path.glob('./*.jpg'))) # counts all ISIC2016 training images
		num_test_img_ISIC2016 = len(list(ISIC2016_test_path.glob('./*.jpg'))) # counts all ISIC2016 test images
		logger.debug('%s %s', "Images available in HAM10000 train dataset:", num_train_img_HAM10000)
		logger.debug('%s %s', "Images available in ISIC2016 train dataset:", num_train_img_ISIC2016)
		logger.debug('%s %s', "Images available in ISIC2016 test dataset:", num_test_img_ISIC2016)

		# HAM10000: Dictionary for Image Names
		imageid_path_dict_HAM10000 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(HAM10000_path, '*.jpg'))}
		# ISIC2016: Dictionary for Image Names
		imageid_path_training_dict_ISIC2016 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2016_training_path, '*.jpg'))}
		imageid_path_test_dict_ISIC2016 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2016_test_path, '*.jpg'))}

		df_HAM10000 = pd.read_csv(str(pathlib.Path.joinpath(pathlib.Path.cwd(), './HAM10000_metadata.csv')))
		df_training_ISIC2016 = pd.read_csv(str(pathlib.Path.joinpath(pathlib.Path.cwd(), './ISIC2016', './ISBI2016_ISIC_Part3_Training_GroundTruth.csv')))
		df_test_ISIC2016 = pd.read_csv(str(pathlib.Path.joinpath(pathlib.Path.cwd(), './ISIC2016', './ISBI2016_ISIC_Part3_Test_GroundTruth.csv')))
		# df = pd.read_pickle(f"../input/skin-cancer-mnist-ham10000-pickle/HAM10000_metadata-h{CFG['img_height']}-w{CFG['img_width']}.pkl")
		pd.set_option('display.max_columns', 500)

		logger.debug("Let's check HAM10000 metadata briefly -> df.head()")
		# logger.debug("Let's check metadata briefly -> df.head()".format(df.head()))
		# print("Let's check metadata briefly -> df.head()")
		display(df_HAM10000.head())

		logger.debug("Let's check ISIC2016 metadata briefly")
		logger.debug("This is ISIC2016 training data")
		df_training_ISIC2016.columns = ['image_id', 'label']
		display(df_training_ISIC2016.head())
		df_test_ISIC2016.columns = ['image_id', 'label']
		logger.debug("This is ISIC2016 test data")
		display(df_test_ISIC2016.head())

		# Given lesion types
		classes_HAM10000 = df_HAM10000.dx.unique() # dx column has labels
		num_classes_HAM10000 = len(classes_HAM10000)
		# self.CFG_num_classes = num_classes
		classes_HAM10000, num_classes_HAM10000

		classes_ISIC2016 = df_training_ISIC2016.label.unique() # second column is label
		num_classes_ISIC2016 = len(classes_ISIC2016)
		classes_ISIC2016, num_classes_ISIC2016

		# Not required for pickled data
		# HAM10000: Creating New Columns for better readability
		df_HAM10000['num_images'] = df_HAM10000.groupby('lesion_id')["image_id"].transform("count")
		df_HAM10000['path'] = df_HAM10000.image_id.map(imageid_path_dict_HAM10000.get)
		df_HAM10000['cell_type'] = df_HAM10000.dx.map(self.lesion_type_dict_HAM10000.get)
		df_HAM10000['cell_type_binary'] = df_HAM10000.dx.map(self.lesion_type_binary_dict_HAM10000.get)
		df_HAM10000['cell_type_idx'] = pd.Categorical(df_HAM10000.dx).codes
		df_HAM10000['cell_type_binary_idx'] = pd.Categorical(df_HAM10000.cell_type_binary).codes
		logger.debug("Let's add some more columns on top of the original metadata for better readability")
		logger.debug("Added columns: 'num_images', 'lesion_id', 'image_id', 'path', 'cell_type'")
		logger.debug("Now, let's show some of records -> df.sample(5)")
		display(df_HAM10000.sample(5))

		df_training_ISIC2016['path'] = df_training_ISIC2016.image_id.map(imageid_path_training_dict_ISIC2016.get)
		df_training_ISIC2016['cell_type_binary'] = df_training_ISIC2016.label.map(self.lesion_type_binary_dict_training_ISIC2016.get)
		df_training_ISIC2016['cell_type_binary_idx'] = pd.Categorical(df_training_ISIC2016.label).codes
		df_test_ISIC2016['path'] = df_test_ISIC2016.image_id.map(imageid_path_test_dict_ISIC2016.get)
		df_test_ISIC2016['cell_type_binary'] = df_test_ISIC2016.label.map(self.lesion_type_binary_dict_test_ISIC2016.get)
		df_test_ISIC2016['cell_type_binary_idx'] = pd.Categorical(df_test_ISIC2016.label).codes
		logger.debug("Let's add some more columns on top of the original metadata for better readability")
		# logger.debug("Added columns: 'num_images', 'lesion_id', 'image_id', 'path', 'cell_type'")
		# logger.debug("Now, let's show some of records -> df.sample(5)")
		display(df_training_ISIC2016.sample(5))
		display(df_test_ISIC2016.sample(5))

		# print("df.shape")
		# display(df.shape)

		# Check null data in metadata
		logger.debug("Check null data in HAM10000 metadata -> df_HAM10000.isnull().sum()")
		display(df_HAM10000.isnull().sum())
		logger.debug("Check null data in ISIC2016 training metadata -> df_training_ISIC2016.isnull().sum()")
		display(df_training_ISIC2016.isnull().sum())
		logger.debug("Check null data in ISIC2016 test metadata -> df_test_ISIC2016.isnull().sum()")
		display(df_test_ISIC2016.isnull().sum())


		# We found there are some null data in age category
		# Filling in with average data
		logger.debug("HAM10000: We found there are some null data in age category. Let's fill them with average data\n")
		logger.debug("df.age.fillna((df_HAM10000.age.mean()), inplace=True) --------------------")
		df_HAM10000.age.fillna((df_HAM10000.age.mean()), inplace=True)


		# Now, we do not have null data
		logger.debug("HAM10000: Let's check null data now -> print(df.isnull().sum())\n")
		logger.debug("HAM10000: There are no null data as below:")
		display(df_HAM10000.isnull().sum())

		# Not required for pickled data
		# resize() order: (width, height)
		img_height = self.image_size[0]
		img_width = self.image_size[1]
		df_HAM10000['image'] = df_HAM10000.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
		df_training_ISIC2016['image'] = df_training_ISIC2016.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
		df_test_ISIC2016['image'] = df_test_ISIC2016.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

		def prepareimages(images):
			# images is a list of images
			images = np.asarray(images).astype(np.float64)
			images = images[:, :, :, ::-1]
			m0 = np.mean(images[:, :, :, 0])
			m1 = np.mean(images[:, :, :, 1])
			m2 = np.mean(images[:, :, :, 2])
			images[:, :, :, 0] -= m0
			images[:, :, :, 1] -= m1
			images[:, :, :, 2] -= m2
			return images
		# df.to_pickle(f"HAM10000_metadata-h{CFG['img_height']}-w{CFG['img_width']}.pkl", compression='infer', protocol=4)

		# print("df.head() -------------------------")
		# pd.options.display.max_columns=300
		
		# pd.options.display.max_columns = None
		# halfdf = df[df.columns[0:len(df.columns)/2]]
		# resthalfdf = df[df.columns[(len(df.columns)/2)+1:len(df.columns)]]
		# halfdf = df.iloc[:, [1,len(df.columns)/2]]
		# halfdf = df.filter(['lesion_id', 'image_id', ])
		# display(halfdf)
		# display(resthalfdf)

		# display(df)

		# Dividing HAM10000 into train/val/test set
		df_single_HAM10000 = df_HAM10000[df_HAM10000.num_images == 1]
		trainset1_HAM10000, testset_HAM10000 = train_test_split(df_single_HAM10000, test_size=0.2,random_state = 80)
		trainset2_HAM10000, validationset_HAM10000 = train_test_split(trainset1_HAM10000, test_size=0.2,random_state = 600)
		trainset3_HAM10000 = df_HAM10000[df_HAM10000.num_images != 1]
		trainset_HAM10000 = pd.concat([trainset2_HAM10000, trainset3_HAM10000])

		# Dividing ISIC2016 into train/val set
		trainset_ISIC2016, validationset_ISIC2016 = train_test_split(df_training_ISIC2016, test_size=0.2,random_state = 80)
		testset_ISIC2016 = df_test_ISIC2016


		# HAM10000 multi-class images/labels
		trainimages_HAM10000 = prepareimages(list(trainset_HAM10000.image))
		testimages_HAM10000 = prepareimages(list(testset_HAM10000.image))
		validationimages_HAM10000 = prepareimages(list(validationset_HAM10000.image))
		trainlabels_HAM10000 = np.asarray(trainset_HAM10000.cell_type_idx)
		testlabels_HAM10000 = np.asarray(testset_HAM10000.cell_type_idx)
		validationlabels_HAM10000 = np.asarray(validationset_HAM10000.cell_type_idx)
		# HAM10000 binary labels
		trainlabels_binary_HAM10000 = np.asarray(trainset_HAM10000.cell_type_binary_idx)
		testlabels_binary_HAM10000 = np.asarray(testset_HAM10000.cell_type_binary_idx)
		validationlabels_binary_HAM10000 = np.asarray(validationset_HAM10000.cell_type_binary_idx)
		
		# ISIC2016 binary images/labels
		trainimages_ISIC2016 = prepareimages(list(trainset_ISIC2016.image))
		testimages_ISIC2016 = prepareimages(list(testset_ISIC2016.image))
		validationimages_ISIC2016 = prepareimages(list(validationset_ISIC2016.image))
		trainlabels_binary_ISIC2016 = np.asarray(trainset_ISIC2016.cell_type_binary_idx)
		testlabels_binary_ISIC2016 = np.asarray(testset_ISIC2016.cell_type_binary_idx)
		validationlabels_binary_ISIC2016 = np.asarray(validationset_ISIC2016.cell_type_binary_idx)

		# height, width 순서
		image_shape = (img_height, img_width, 3)

		# Unpack all image pixels using asterisk(*) with dimension (shape[0])
		trainimages_HAM10000 = trainimages_HAM10000.reshape(trainimages_HAM10000.shape[0], *image_shape)
		trainimages_ISIC2016 = trainimages_ISIC2016.reshape(trainimages_ISIC2016.shape[0], *image_shape)

		data_gen_HAM10000 = ImageDataGenerator(
			rotation_range = 90,    # randomly rotate images in the range (degrees, 0 to 180)
			zoom_range = 0.1,            # Randomly zoom image 
			width_shift_range = 0.1,   # randomly shift images horizontally
			height_shift_range = 0.1,  # randomly shift images vertically
			horizontal_flip= False,              # randomly flip images
			vertical_flip= False                 # randomly flip images
		)
		data_gen_ISIC2016 = ImageDataGenerator(
			rotation_range = 90,    # randomly rotate images in the range (degrees, 0 to 180)
			zoom_range = 0.1,            # Randomly zoom image 
			width_shift_range = 0.1,   # randomly shift images horizontally
			height_shift_range = 0.1,  # randomly shift images vertically
			horizontal_flip= False,              # randomly flip images
			vertical_flip= False                 # randomly flip images
		)
		data_gen_HAM10000.fit(trainimages_HAM10000)
		data_gen_ISIC2016.fit(trainimages_ISIC2016)

		HAM10000_multiclass = (trainimages_HAM10000, testimages_HAM10000, validationimages_HAM10000, trainlabels_HAM10000, testlabels_HAM10000, validationlabels_HAM10000, num_classes_HAM10000)
		HAM10000_binaryclass = (trainimages_HAM10000, testimages_HAM10000, validationimages_HAM10000, trainlabels_binary_HAM10000, testlabels_binary_HAM10000, validationlabels_binary_HAM10000, 2)
		ISIC2016_binaryclass = (trainimages_ISIC2016, testimages_ISIC2016, validationimages_ISIC2016, trainlabels_binary_ISIC2016, testlabels_binary_ISIC2016, validationlabels_binary_ISIC2016, num_classes_ISIC2016)

		return data_gen_HAM10000, HAM10000_multiclass, HAM10000_binaryclass, data_gen_ISIC2016, ISIC2016_binaryclass
	
	def combine_images(self, **kwargs):
		# combined = np.array(0, )
		for idx, (key, value) in enumerate(kwargs.items()):
			print("Combining: " + key)
			if idx==0:
				combined = value
			elif idx>0:
				combined = np.vstack((combined, value))
			# combined = np.concatenate(value)
		return combined
	
	def combine_labels(self, **kwargs):
		for idx, (key, value) in enumerate(kwargs.items()):
			print("Combining: " + key)
			if idx==0:
				combined = value
			elif idx>0:
				combined = np.hstack((combined, value))
			# combined = np.concatenate(value)
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


	def loadTestData(self):
		# test_data_dir = pathlib.Path(path)
		num_test_img = len(list(self.test_data_dir.glob('*/*.jpg'))) # counts all images inside 'Test' folder
		print("Images available in test dataset:", num_test_img)



	

	

