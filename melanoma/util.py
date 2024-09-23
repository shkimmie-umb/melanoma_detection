from glob import glob
import os
import pathlib
import torch
import torchvision

import io

import melanoma as mel
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler
# from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

import itertools

class Util:
	def __init__(self):
		pass
	
	@staticmethod
	def contains_images(folder_path):
		valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
		if os.path.isdir(folder_path):
			return any(f.lower().endswith(valid_extensions) for f in os.listdir(folder_path))
		return False

	@staticmethod
	def loadDatasetFromDirectory(path, preprocessing):
		# path: A directory where 'Train', 'Val', 'Test' folders exist

		image_folders = {
			'Train': None,
			'Val': None
		}

		# image_folders = {x: torchvision.datasets.ImageFolder(os.path.join(path, x),
		# 										preprocessing[x])
		# 				for x in ['Train', 'Val'] if os.path.isdir(os.path.join(path, x))}
		for x in ['Train', 'Val']:
			folder_path = os.path.join(path, x)
			if os.path.isdir(folder_path):
				image_folders[x] = torchvision.datasets.ImageFolder(folder_path, preprocessing[x])


		return image_folders


	@staticmethod
	def combineDatasets(*args, preprocessing):
		from collections import defaultdict
		
		dataloaders = defaultdict(list)
		image_folders = defaultdict(list)
		combined_data = defaultdict(list)
		num_classes = defaultdict(list)

		print('Combining...')
		for idx, db_path in enumerate(args[0]):
			dbname = pathlib.Path(db_path).parts[-2]
			
			print(f'Combining {idx+1}th db out of {len(args[0])} dbs')
			image_folder = Util.loadDatasetFromDirectory(path=db_path, preprocessing=preprocessing)

			image_folders['Train'].append(image_folder['Train'])
			
			
			
			if image_folder['Val'] is not None:
				image_folders['Val'].append(image_folder['Val'])

				assert len(image_folder['Train']) + len(image_folder['Val']) == \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['trainimages'] + \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['validationimages']
				assert image_folder['Train'].classes == mel.Parser.classes_melanoma_binary
				assert image_folder['Val'].classes == mel.Parser.classes_melanoma_binary
			elif image_folder['Val'] is None:
				assert len(image_folder['Train']) == \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['trainimages'] + \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['validationimages']
				assert image_folder['Train'].classes == mel.Parser.classes_melanoma_binary
			
			



		print('Stacking data')
		for idx, phase in enumerate(list(image_folders.keys())):
			combined_data[phase] = torch.utils.data.ConcatDataset(image_folders[phase])

		dataloaders = {}
		dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
			sampler=ImbalancedDatasetSampler(combined_data['Train']), shuffle=False, num_workers=4)
		# dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
		# 	shuffle=True, num_workers=32, pin_memory=True)
		# dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
		# 	shuffle=False, num_workers=4)
		dataloaders['Val'] = torch.utils.data.DataLoader(combined_data['Val'], batch_size=32,
													shuffle=True, num_workers=4)


		dataset_sizes = {x: len(combined_data[x]) for x in ['Train', 'Val']}
		
		print('Combining complete')

		return dataloaders, dataset_sizes




	

	

