import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import Model

class Util:
	trainDataPath = ''
	testDataPath = ''
	train_ds = ''
	val_ds = ''
	class_names = []
	def __init__(self):
		self.trainDataPath = ''
		self.testDataPath = ''


	def loadTrainData(self, path, seed_val=1, split_portion=0.2, image_size=(256, 256), batch_size=32, color_mode='rgb'):
		print("path: ", path)
		print("seed value: ", seed_val)
		print("color_mode: ", color_mode)
		self.trainDataPath = pathlib.Path(path)
		num_train_img = len(list(self.trainDataPath.glob('*/*.jpg'))) # counts all images inside 'Train' folder
		print("Images available in train dataset:", num_train_img)

		# Loading the training data
		# using seed=123 while creating dataset using tf.keras.preprocessing.image_dataset_from_directory
		# resizing images to the size img_height*img_width, while writing the dataset
		self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(self.trainDataPath,
		                                                               seed=seed_val,
		                                                               validation_split=split_portion,
		                                                               image_size=image_size,
		                                                               batch_size=batch_size,
		                                                               color_mode=color_mode,
		                                                               subset='training')
		train_ds = self.train_ds

		self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(self.trainDataPath,
		                                                               seed=seed_val,
		                                                               validation_split=split_portion,
		                                                               image_size=image_size,
		                                                               batch_size=batch_size,
		                                                               color_mode=color_mode,
		                                                               subset='validation')
		val_ds = self.val_ds
		# List out all the classes of training data in a list.
		self.class_names = self.train_ds.class_names
		print("Training classes are: ")
		class_names = self.class_names
		print(class_names)

		return train_ds, val_ds, class_names

	def loadTestData(self,path):
		test_data_dir = pathlib.Path(path)
		num_test_img = len(list(test_data_dir.glob('*/*.jpg'))) # counts all images inside 'Test' folder
		print("Images available in test dataset:", num_test_img)

	def data_viewer(self):
		### visualize one instance of all the nine classes present in the dataset
		plt.figure(figsize=(15,15))
		for i in range(len(self.class_names)):
			plt.subplot(3,3,i+1)
			image= plt.imread(str(list(self.trainDataPath.glob(self.class_names[i]+'/*.jpg'))[0]))
			plt.title(self.class_names[i])
			plt.imshow(image)

	def visualize_cnn(history, epochs):
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		
		epochs_range = range(epochs)
		
		plt.figure(figsize=(12, 8))
		plt.subplot(1, 2, 1)
		plt.plot(epochs_range, acc, label='Training Accuracy')
		plt.plot(epochs_range, val_acc, label='Validation Accuracy')
		plt.legend(loc='upper left')
		plt.title('Training and Validation Accuracy')
		
		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, loss, label='Training Loss')
		plt.plot(epochs_range, val_loss, label='Validation Loss')
		plt.legend(loc='upper center')
		plt.title('Training and Validation Loss')
		plt.show()

	def trainData(self, train_ds_input, val_ds_input, epochs, class_names):
		# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB).
		# The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images.
		# `Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch.
		# `Dataset.prefetch()` overlaps data preprocessing and model execution while training.


		AUTOTUNE = tf.data.experimental.AUTOTUNE
		train_ds = train_ds_input.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
		val_ds = val_ds_input.cache().prefetch(buffer_size=AUTOTUNE)
		cnnmd1 = Model.Model()
		img_width = 180
		img_height = 180
		##ToDo: change img size passing logic
		model = cnnmd1.CNN(img_width, img_height, class_names) # Get CNN model to use
		# Compiling the model
		model.compile(optimizer='adam',
		              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		              metrics=['accuracy'])
		model.summary()

		# Training the model
		# epochs = 20
		history = model.fit(
		  train_ds,
		  validation_data=val_ds,
		  epochs=epochs
		)
		return history, epochs

