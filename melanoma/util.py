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
	def loadDatasetFromDirectory(path, preprocessing):
		# path: A directory where 'Train', 'Val', 'Test' folders exist

		
		image_folders = {x: torchvision.datasets.ImageFolder(os.path.join(path, x),
												preprocessing[x])
						for x in ['Train', 'Val', 'Test']}

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
			
			image_folder = Util.loadDatasetFromDirectory(path=db_path, preprocessing=preprocessing)
			print(f'Combining {idx+1}th db out of {len(args[0])} dbs')

			image_folders['Train'].append(image_folder['Train'])
			image_folders['Val'].append(image_folder['Val'])
			image_folders['Test'].append(image_folder['Test'])

			dbname = pathlib.Path(db_path).parts[-2]

			assert len(image_folder['Train']) + len(image_folder['Val']) == \
			mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['trainimages'] + \
			mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['validationimages']

			assert image_folder['Train'].classes == mel.Parser.classes_melanoma_binary
			if len(image_folder['Val']) > 0:
				assert image_folder['Val'].classes == mel.Parser.classes_melanoma_binary
			if len(image_folder['Test']) > 0:
				assert image_folder['Test'].classes == mel.Parser.classes_melanoma_binary

			# datachecker.append(mel.CommonData().dbNumImgs[mel.DatasetType.db_path.stem]['testimages'])

			
		# 	if len(validationdata['validationimages']) > 0:
		# 		assert len(validationdata['validationimages']) == len(validationdata['validationlabels']) and \
		# 			len(validationdata['validationlabels']) == len(validationdata['validationids'])
		# 		combined_data['validationimages'].append(validationdata['validationimages'])
		# 		combined_data['validationlabels'].append(validationdata['validationlabels'])
		# 		combined_data['validationids'].append(validationdata['validationids'])



		print('Stacking data')
		for idx, phase in enumerate(list(image_folders.keys())):
			combined_data[phase] = torch.utils.data.ConcatDataset(image_folders[phase])

		dataloaders = {}
		dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
			sampler=ImbalancedDatasetSampler(combined_data['Train']), shuffle=False, num_workers=4)
		# dataloaders['Train'] = torch.utils.data.DataLoader(combined_data['Train'], batch_size=32, 
		# 	shuffle=False, num_workers=4)
		dataloaders['Val'] = torch.utils.data.DataLoader(combined_data['Val'], batch_size=32,
													shuffle=True, num_workers=4)
		dataloaders['Test'] = torch.utils.data.DataLoader(combined_data['Test'], batch_size=32,
													shuffle=False)

		dataset_sizes = {x: len(combined_data[x]) for x in ['Train', 'Val', 'Test']}
		
		# Data Validation
		# datasetsizes = defaultdict(list)
		# for i, key in enumerate(combined_data.keys()):
		# 	datasetsizes[key] = combined_data[key].cumulative_sizes[-1]
		# 	assert datasetsizes[key] == 
		# 	for j in range(len(combined_data[key].datasets)):
		# 		assert combined_data[key].datasets[j].classes == mel.Parser.classes_melanoma_binary

		# dbname = pathlib.Path(path).parts[-2]

		# assert dataset_sizes['Train'] + dataset_sizes['Val'] == \
		# 	mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['trainimages'] + \
		# 	mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['validationimages']
		# assert dataset_sizes['Test'] == mel.CommonData().dbNumImgs[mel.DatasetType[dbname]]['testimages']
		# assert class_names == mel.Parser.classes_melanoma_binary

		# combined_data['trainimages'] = np.vstack(combined_data['trainimages'])
		# combined_data['trainlabels'] = np.vstack(combined_data['trainlabels'])
		# combined_data['trainids'] = np.vstack(combined_data['trainids'])
		# combined_data['validationimages'] = np.vstack(combined_data['validationimages'])
		# combined_data['validationlabels'] = np.vstack(combined_data['validationlabels'])
		# combined_data['validationids'] = np.vstack(combined_data['validationids'])

		# assert len(combined_data['trainimages']) == len(combined_data['trainlabels']) \
		# and len(combined_data['trainlabels']) == len(combined_data['trainids'])
		# assert len(combined_data['validationimages']) == len(combined_data['validationlabels']) \
		# and len(combined_data['validationlabels']) == len(combined_data['validationids'])
		
		print('Combining complete')

		return dataloaders, dataset_sizes




	

	

