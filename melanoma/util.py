import pathlib
import tensorflow as tf


class Util:
	# trainDataPath = ''
	# testDataPath = ''
	# train_ds = ''
	# val_ds = ''
	# class_names = []
	def __init__(self, path, image_size=(None, None), seed_val=1, split_portion=0.2, batch_size=32, color_mode='rgb'):
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

	def loadTestData(self):
		# test_data_dir = pathlib.Path(path)
		num_test_img = len(list(self.test_data_dir.glob('*/*.jpg'))) # counts all images inside 'Test' folder
		print("Images available in test dataset:", num_test_img)



	

	

