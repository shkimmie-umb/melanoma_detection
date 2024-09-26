from glob import glob
import os
import pathlib
import torch
import torchvision
import torchtune
from collections import defaultdict

import io
import numpy as np

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
	def loadDatasetFromDirectory_fast(path, pre_transform, post_transform):
		# path: A directory where 'Train', 'Val', 'Test' folders exist

		data = {
			'Train': None,
			'Val': None
		}

		# image_folders = {x: torchvision.datasets.ImageFolder(os.path.join(path, x),
		# 										preprocessing[x])
		# 				for x in ['Train', 'Val'] if os.path.isdir(os.path.join(path, x))}
		for x in ['Train', 'Val']:
			folder_path = os.path.join(path, x)
			if os.path.isdir(folder_path):
				data[x] = mel.DataLoaderFast(folder_path, pre_transform[x], post_transform[x])


		return data


	@staticmethod
	def combineDatasets(*args, preprocessing):
		
		
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
			# combined_data[phase] = torchtune.datasets.ConcatDataset(image_folders[phase])

		dataloaders = {}
		dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
			sampler=ImbalancedDatasetSampler(combined_data['Train']), shuffle=False, pin_memory=True,
			num_workers=4, prefetch_factor = 2, drop_last=True)
		# dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
		# 	shuffle=True, num_workers=32, pin_memory=True)
		# dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
		# 	shuffle=False, num_workers=4)
		dataloaders['Val'] = torch.utils.data.DataLoader(combined_data['Val'], batch_size=32,
													shuffle=True, pin_memory=True,
													num_workers=4, prefetch_factor=2)


		dataset_sizes = {x: len(combined_data[x]) for x in ['Train', 'Val']}
		
		print('Combining complete')

		return dataloaders, dataset_sizes
	
	@staticmethod
	def combineDatasets_hdf5(*args):
		combined_data = defaultdict(list)

		print('Combining...')
		for idx, h5_file in enumerate(args[0]):
			
			traindata, validationdata, testdata = mel.Parser.open_H5(h5_file)
			print(f'Combining {idx+1}th db out of {len(args[0])} dbs')

			combined_data['trainimages'].append(traindata['trainimages'])
			combined_data['trainlabels'].append(traindata['trainlabels'])
			combined_data['trainids'].append(traindata['trainids'])

			
			if len(validationdata['validationimages']) > 0:
				assert len(validationdata['validationimages']) == len(validationdata['validationlabels']) and \
					len(validationdata['validationlabels']) == len(validationdata['validationids'])
				combined_data['validationimages'].append(validationdata['validationimages'])
				combined_data['validationlabels'].append(validationdata['validationlabels'])
				combined_data['validationids'].append(validationdata['validationids'])



		print('Stacking data')
		combined_data['trainimages'] = np.vstack(combined_data['trainimages'])
		combined_data['trainlabels'] = np.vstack(combined_data['trainlabels'])
		combined_data['trainids'] = np.vstack(combined_data['trainids'])
		combined_data['validationimages'] = np.vstack(combined_data['validationimages'])
		combined_data['validationlabels'] = np.vstack(combined_data['validationlabels'])
		combined_data['validationids'] = np.vstack(combined_data['validationids'])

		assert len(combined_data['trainimages']) == len(combined_data['trainlabels']) \
		and len(combined_data['trainlabels']) == len(combined_data['trainids'])
		assert len(combined_data['validationimages']) == len(combined_data['validationlabels']) \
		and len(combined_data['validationlabels']) == len(combined_data['validationids'])
		
		print('Combining complete')

		return combined_data
	

	@staticmethod
	def combineDatasets_fast(*args, pre_transform, post_transform):
		
		
		dataloaders = defaultdict(list)
		data = defaultdict(list)
		combined_data = defaultdict(list)
		num_classes = defaultdict(list)

		print('Combining...')
		for idx, db_path in enumerate(args[0]):
			dbname = pathlib.Path(db_path).parts[-2]
			
			print(f'Combining {idx+1}th db out of {len(args[0])} dbs')
			datum = Util.loadDatasetFromDirectory_fast(path=db_path, pre_transform=pre_transform, post_transform=post_transform)

			data['Train'].append(datum['Train'])
			
			
			
			if datum['Val'] is not None:
				data['Val'].append(datum['Val'])

				assert len(datum['Train']) + len(datum['Val']) == \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['trainimages'] + \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['validationimages']
				# assert datum['Train'].classes == mel.Parser.classes_melanoma_binary
				# assert datum['Val'].classes == mel.Parser.classes_melanoma_binary
			elif datum['Val'] is None:
				assert len(datum['Train']) == \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['trainimages'] + \
				mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['validationimages']
				# assert datum['Train'].classes == mel.Parser.classes_melanoma_binary
			
			



		print('Stacking data')
		for idx, phase in enumerate(list(data.keys())):
			combined_data[phase] = torch.utils.data.ConcatDataset(data[phase])
			# combined_data[phase] = torchtune.datasets.ConcatDataset(image_folders[phase])

		dataloaders = {}
		dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
			sampler=ImbalancedDatasetSampler(combined_data['Train']), shuffle=False, pin_memory=True,
			num_workers=4, prefetch_factor=2, drop_last=True)
		# dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
		# 	shuffle=True, num_workers=32, pin_memory=True)
		# dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
		# 	shuffle=False, num_workers=4)
		dataloaders['Val'] = torch.utils.data.DataLoader(combined_data['Val'], batch_size=32,
													shuffle=True, pin_memory=True)


		dataset_sizes = {x: len(combined_data[x]) for x in ['Train', 'Val']}
		
		print('Combining complete')

		return dataloaders, dataset_sizes



	

	

