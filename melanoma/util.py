from glob import glob
import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from enum import Enum

import PIL
from PIL import Image

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import matplotlib.pyplot as plt


from IPython.core.debugger import Pdb
from IPython.display import display

import logging

class DatasetType(Enum):
    HAM10000 = 1


class Util:
	# trainDataPath = ''
	# testDataPath = ''
	# train_ds = ''
	# val_ds = ''
	# class_names = []
	def __init__(self, path, image_size=(None, None), seed_val=1, split_portion=0.2, batch_size=32, color_mode='rgb'):
		self.base_dir = pathlib.Path(path)
		self.trainDataPath = pathlib.Path(path+'/Train')
		self.testDataPath = pathlib.Path(path+'/Test')
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
		self.lesion_type_dict = {
			'bkl'  : 'Pigmented Benign keratosis',
			'nv'   : 'Melanocytic nevi', # nevus
			'df'   : 'Dermatofibroma',
			'mel'  : 'Melanoma',
			'vasc' : 'Vascular lesions',
			'bcc'  : 'Basal cell carcinoma',
			'akiec': 'Actinic keratoses',
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

	


	
	def prepareimages(self, images):
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
	
	def loadMelanomaDataset(self, mode):
		# create logger
		logger = logging.getLogger('HAM10000 classification example')
		logger.setLevel(logging.DEBUG)

		# # create console handler and set level to debug
		# ch = logging.StreamHandler()
		# ch.setLevel(logging.DEBUG)

		# # add ch to logger
		# logger.addHandler(ch)




		if mode == DatasetType.HAM10000:
			# Set default image size for HAM10000
			self.image_size = (112, 150) # height, width
			logger.debug('%s %s', "path: ", self.base_dir)
			logger.debug('%s %s', "seed value: ", self.seed_val)
			logger.debug('%s %s', "color_mode: ", self.color_mode)
			# print("path: ", self.base_dir)
			# print("seed value: ", self.seed_val)
			# print("color_mode: ", self.color_mode)
			num_train_img = len(list(self.base_dir.glob('./*.jpg'))) # counts all images
			logger.debug('%s %s', "Images available in train dataset:", num_train_img)

			#Dictionary for Image Names
			base_skin_dir = './HAM10000_images_combined'
			# Pdb().set_trace()
			imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_skin_dir, '*.jpg'))}

			df = pd.read_csv('./HAM10000_metadata.csv')
			# df = pd.read_pickle(f"../input/skin-cancer-mnist-ham10000-pickle/HAM10000_metadata-h{CFG['img_height']}-w{CFG['img_width']}.pkl")
			pd.set_option('display.max_columns', 500)

			logger.debug("Let's check metadata briefly -> df.head()")
			# logger.debug("Let's check metadata briefly -> df.head()".format(df.head()))
			# print("Let's check metadata briefly -> df.head()")
			display(df.head())
			

			# Given lesion types
			classes = df.dx.unique()
			num_classes = len(classes)
			self.CFG_num_classes = num_classes
			classes, num_classes



			# Not required for pickled data
			# Creating New Columns for better readability

			df['num_images'] = df.groupby('lesion_id')["image_id"].transform("count")
			df['path'] = df.image_id.map(imageid_path_dict.get)
			df['cell_type'] = df.dx.map(self.lesion_type_dict.get)
			df['cell_type_idx'] = pd.Categorical(df.dx).codes
			logger.debug("Let's add some more columns on top of the original metadata for better readability")
			logger.debug("Added columns: 'num_images', 'lesion_id', 'image_id', 'path', 'cell_type'")
			logger.debug("Now, let's show some of records -> df.sample(5)")
			display(df.sample(5))

			# print("df.shape")
			# display(df.shape)

			# Check null data in metadata
			logger.debug("Check null data in metadata -> df.isnull().sum()")
			display(df.isnull().sum())

			# We found there are some null data in age category
			# Filling in with average data
			logger.debug("We found there are some null data in age category. Let's fill them with average data\n")
			logger.debug("df.age.fillna((df.age.mean()), inplace=True) --------------------")
			df.age.fillna((df.age.mean()), inplace=True)


			# Now, we do not have null data
			logger.debug("Let's check null data now -> print(df.isnull().sum())\n")
			logger.debug("There are no null data as below:")
			display(df.isnull().sum())

			# Not required for pickled data
			# resize(width, height 순서)
			img_height = self.image_size[0]
			img_width = self.image_size[1]
			df['image'] = df.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
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

			# Dividing into train/val/test set
			df_single = df[df.num_images == 1]
			trainset1, testset = train_test_split(df_single, test_size=0.2,random_state = 80)
			trainset2, validationset = train_test_split(trainset1, test_size=0.2,random_state = 600)
			trainset3 = df[df.num_images != 1]
			trainset = pd.concat([trainset2, trainset3])

			# Pdb().set_trace()
			trainimages = self.prepareimages(list(trainset.image))
			testimages = self.prepareimages(list(testset.image))
			validationimages = self.prepareimages(list(validationset.image))
			trainlabels = np.asarray(trainset.cell_type_idx)
			testlabels = np.asarray(testset.cell_type_idx)
			validationlabels = np.asarray(validationset.cell_type_idx)

			# height, width 순서
			image_shape = (img_height, img_width, 3)

			# Unpack all image pixels using asterisk(*) with dimension (shape[0])
			trainimages = trainimages.reshape(trainimages.shape[0], *image_shape)

			data_gen = ImageDataGenerator(
				rotation_range = 90,    # randomly rotate images in the range (degrees, 0 to 180)
				zoom_range = 0.1,            # Randomly zoom image 
				width_shift_range = 0.1,   # randomly shift images horizontally
				height_shift_range = 0.1,  # randomly shift images vertically
				horizontal_flip= False,              # randomly flip images
				vertical_flip= False                 # randomly flip images
			)
			data_gen.fit(trainimages)

			return data_gen, trainimages, testimages, validationimages, trainlabels, testlabels, validationlabels, num_classes

	


	def loadTestData(self):
		# test_data_dir = pathlib.Path(path)
		num_test_img = len(list(self.test_data_dir.glob('*/*.jpg'))) # counts all images inside 'Test' folder
		print("Images available in test dataset:", num_test_img)



	

	

