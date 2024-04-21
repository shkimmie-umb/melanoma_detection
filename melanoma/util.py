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
import io
import h5py


from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img, array_to_img
from .preprocess import Preprocess

import itertools
from itertools import combinations

import matplotlib.pyplot as plt


from IPython.core.debugger import Pdb
from IPython.display import display

import logging
import melanoma as mel


class ClassType(Enum):
	multi = 1
	binary = 2

class Util:
	def __init__(self, path, square_size, image_size=(None, None)):
		self.base_dir = pathlib.Path(path)
		
		# self.trainDataPath = pathlib.Path.joinpath(path, '/Train')
		# self.testDataPath = pathlib.Path.joinpath(path, '/Test')
		# self.seed_val = seed_val
		# self.split_portion = split_portion
		self.square_size = square_size
		self.image_size = image_size # (height, width)
		# self.batch_size = batch_size
		# self.color_mode = color_mode
		# self.epochs = epochs
		self.class_names = []
		# self.class_names = class_names
		self.train_ds = ''
		self.val_ds = ''

		

		self.foldersExist = ''
		self.RGBfolders = ''



	

	
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
		

		
		# def ExtractPixel(image):
		# 			return [item[1] for item in image]
		
		

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
		


		# HAM10000 multi-class images/labels
		if datasettype.value == mel.DatasetType.HAM10000.value:
			pass



		
		if datasettype.value == mel.DatasetType.ISIC2016.value:
			pass


		if datasettype.value == mel.DatasetType.ISIC2017.value:
			pass


		if datasettype.value == mel.DatasetType.ISIC2018.value:
			pass
			

		if datasettype.value == mel.DatasetType.ISIC2019.value:
			pass
			

		if datasettype.value == mel.DatasetType.ISIC2020.value:
			pass
			

		if datasettype.value == mel.DatasetType.PH2.value:
			pass
			


		if datasettype.value == mel.DatasetType._7_point_criteria.value:
			pass
			

		if datasettype.value == mel.DatasetType.PAD_UFES_20.value:
			pass
			

		if datasettype.value == mel.DatasetType.KaggleMB.value:

			pass

		if datasettype.value == mel.DatasetType.MEDNODE.value:
			pass
				

			


		

		



		
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
	
	def combineDatasets(self, *args):
		
		trainimgs_list = []
		valimgs_list = []
		trainlabels_list = []
		vallabels_list = []

		print('Combining...')
		for idx, pkl_file in enumerate(args[0]):
			
			datum = pickle.load(open(pkl_file, 'rb'))
			print(f'Combining {idx+1} db out of {len(args[0])} dbs')
			# [0]:Trainimgs, [1]:Testimgs, [2]:Valimgs, [3]:trainlabels, [4]:Testlabels, [5]:Vallabels
			trainimgs_list.append(datum[0])
			if datum[2] is not None:
				valimgs_list.append(datum[2])
			trainlabels_list.append(datum[3])
			if datum[5] is not None:
				vallabels_list.append(datum[5])
			

		
		
		assert sum(list(map(lambda v: len(v), trainimgs_list))) == sum(list(map(lambda v: len(v), trainlabels_list)))
		for idx, file in enumerate(trainimgs_list):
			assert trainimgs_list[idx].shape[0] == trainlabels_list[idx].shape[0]
		
		
		print('Stacking training images')
		list_size = sum(list(map(lambda v: len(v), trainimgs_list)))
		trainimgs_list = np.vstack(trainimgs_list)
		assert trainimgs_list.shape[0] == list_size
		

		print('Stacking training labels')
		list_size = sum(list(map(lambda v: len(v), trainlabels_list)))
		trainlabels_list = np.vstack(trainlabels_list)
		assert trainlabels_list.shape[0] == list_size
		
		

		assert sum(list(map(lambda v: len(v), valimgs_list))) == sum(list(map(lambda v: len(v), vallabels_list)))
		for idx, file in enumerate(valimgs_list):
			assert valimgs_list[idx].shape[0] == vallabels_list[idx].shape[0]
		
		

		print('Stacking validation images')
		list_size = sum(list(map(lambda v: len(v), valimgs_list)))
		valimgs_list = np.vstack(valimgs_list)		
		assert valimgs_list.shape[0] == list_size
		

		print('Stacking validation labels')
		list_size = sum(list(map(lambda v: len(v), vallabels_list)))
		vallabels_list = np.vstack(vallabels_list)
		assert vallabels_list.shape[0] == list_size

		return trainimgs_list, None, valimgs_list, trainlabels_list, None, vallabels_list

	def combineSavedDatasetsToFile(self, new_path, new_filename, *args):
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
			# [0]:Trainimgs, [1]:Testimgs, [2]:Valimgs, [3]:trainlabels, [4]:Testlabels, [5]:Vallabels
			trainimgs_list.append(datum[0])
			if datum[2] is not None:
				valimgs_list.append(datum[2])
			trainlabels_list.append(datum[3])
			if datum[5] is not None:
				vallabels_list.append(datum[5])
			

		
		
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



	

	

