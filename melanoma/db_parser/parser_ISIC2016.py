from .parser import *


class parser_ISIC2016(Parser):

    def __init__(self, base_dir, square_size, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        # ISIC2016
        self.lesion_type_binary_dict_training_ISIC2016 = {
            'benign' : 'Non-Melanoma',
            'malignant' : 'Melanoma',
        }
        self.lesion_type_binary_dict_test_ISIC2016 = {
            0.0 : 'Non-Melanoma',
            1.0 : 'Melanoma',
        }


    def saveDatasetToFile(self, augment_ratio=None):
        
        
        datasetname = mel.DatasetType.ISIC2016.name

        self.makeFolders(datasetname)
        

        

        ISIC2016_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Training_Data')
        ISIC2016_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './ISIC2016', './ISBI2016_ISIC_Part3_Test_Data')

        num_train_img_ISIC2016 = len(list(ISIC2016_training_path.glob('./*.jpg'))) # counts all ISIC2016 training images
        num_test_img_ISIC2016 = len(list(ISIC2016_test_path.glob('./*.jpg'))) # counts all ISIC2016 test images

        assert num_train_img_ISIC2016 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['trainimages']
        assert num_test_img_ISIC2016 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['testimages']

        self.logger.debug('%s %s', "Images available in ISIC2016 train dataset:", num_train_img_ISIC2016)
        self.logger.debug('%s %s', "Images available in ISIC2016 test dataset:", num_test_img_ISIC2016)

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

        self.logger.debug("Let's check ISIC2016 metadata briefly")
        self.logger.debug("This is ISIC2016 training data")
        display(df_training_ISIC2016.head())
        self.logger.debug("This is ISIC2016 test data")
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
        self.logger.debug("ISIC2016 training df")
        display(df_training_ISIC2016.sample(10))
        self.logger.debug("ISIC2016 test df")
        display(df_test_ISIC2016.sample(10))

        self.logger.debug("Check null data in ISIC2016 training metadata -> df_training_ISIC2016.isnull().sum()")
        display(df_training_ISIC2016.isnull().sum())
        self.logger.debug("Check null data in ISIC2016 test metadata -> df_test_ISIC2016.isnull().sum()")
        display(df_test_ISIC2016.isnull().sum())


        df_training_ISIC2016['image'] = df_training_ISIC2016.path.map(
        lambda x:(
            img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                         resize_width=self.resize_width, resize_height=self.resize_height)),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )


        df_test_ISIC2016['image'] = df_test_ISIC2016.path.map(
        lambda x:(
            img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                resize_width=self.resize_width, resize_height=self.resize_height)),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )



        assert all(df_training_ISIC2016.cell_type_binary.unique() == df_test_ISIC2016.cell_type_binary.unique())
        labels = df_training_ISIC2016.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)
        if not self.isWholeFeatureExist or not self.isTrainFeatureExist or not self.isValFeatureExist or not self.isTestFeatureExist:
            for i in labels:
                os.makedirs(f"{self.whole_feature_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_feature_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_feature_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_feature_folder}/{i}", exist_ok=True)

        # Dividing ISIC2016 into train/val set
        
        trainset_ISIC2016, validationset_ISIC2016 = train_test_split(df_training_ISIC2016, test_size=self.split_ratio, random_state = self.pseudo_num)
        # ISIC2016 test data is given, so there is no need to create test dataset separately
        testset_ISIC2016 = df_test_ISIC2016


        self.preprocessor.saveNumpyImagesToFiles(trainset_ISIC2016, df_training_ISIC2016, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(validationset_ISIC2016, df_training_ISIC2016, self.val_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(testset_ISIC2016, df_test_ISIC2016, self.test_rgb_folder)


        # ISIC2016 binary images/labels
        # trainpixels_ISIC2016 = []; testpixels_ISIC2016 = []; validationpixels_ISIC2016 = []
        # trainids = []; testids = []; validationids = []
        # trainlabels_binary_ISIC2016 = []; testlabels_binary_ISIC2016 = []; validationlabels_binary_ISIC2016 = []

        # for idx, obj in trainset_ISIC2016.iterrows():
        #     img_obj = obj['image']
        #     trainpixels_ISIC2016.append(img_obj[1])
        #     trainids.append(img_obj[2].stem)
        # for idx, img_obj in enumerate(testset_ISIC2016['image']):
        #     testpixels_ISIC2016.append(img_obj[1])
        #     testids.append(img_obj[2].stem)
        # for idx, img_obj in enumerate(validationset_ISIC2016['image']):
        #     validationpixels_ISIC2016.append(img_obj[1])
        #     validationids.append(img_obj[2].stem)
        trainpixels_ISIC2016 = list(map(lambda x:x[0], trainset_ISIC2016['image'])) # Filter out only pixel from the list
        testpixels_ISIC2016 = list(map(lambda x:x[0], testset_ISIC2016['image']))
        validationpixels_ISIC2016 = list(map(lambda x:x[0], validationset_ISIC2016['image']))

        trainids = list(map(lambda x:x[1].stem, trainset_ISIC2016['image'])) # Filter out only pixel from the list
        testids = list(map(lambda x:x[1].stem, testset_ISIC2016['image']))
        validationids = list(map(lambda x:x[1].stem, validationset_ISIC2016['image']))

        # trainimages_ISIC2016 = preprocessor.normalizeImgs(imgs=trainpixels_ISIC2016, networktype=networktype, 
        # 										 uniform_normalization=uniform_normalization)
        # validationimages_ISIC2016 = preprocessor.normalizeImgs(imgs=validationpixels_ISIC2016, networktype=networktype,
        # 											  uniform_normalization=uniform_normalization)
        # testimages_ISIC2016 = preprocessor.normalizeImgs(imgs=testpixels_ISIC2016, networktype=networktype,
        # 										uniform_normalization=uniform_normalization)

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
        # assert trainimages_ISIC2016.shape[0] == trainlabels_binary_ISIC2016.shape[0]
        # assert validationimages_ISIC2016.shape[0] == validationlabels_binary_ISIC2016.shape[0]
        # assert testimages_ISIC2016.shape[0] == testlabels_binary_ISIC2016.shape[0]

        # trainimages_ISIC2016 = trainimages_ISIC2016.reshape(trainimages_ISIC2016.shape[0], *image_shape)

        # Feature saving
        # for idx, order in enumerate(testset_ISIC2016.index):
        # 	img = array_to_img(testimages_ISIC2016[idx])
        # 	label = testset_ISIC2016.cell_type_binary[order]
        # 	assert label == df_test_ISIC2016.cell_type_binary[order]
        # 	img.save(f"{test_feature_folder}/{label}/{testset_ISIC2016.image[order][2].stem}.jpg", quality=100, subsampling=0)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width

        # assert len(trainimages_bytes) + len(validationimages_bytes) == 900
        # assert len(trainimages_bytes) == df_training_ISIC2016.shape[0]*(1-split_ratio)
        # assert len(validationimages_bytes) == df_training_ISIC2016.shape[0]*split_ratio
        # assert len(testimages_bytes) == 379
        # create HDF5 file
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_ISIC2016, 
                        testpxs=testpixels_ISIC2016, 
                        validationpxs=validationpixels_ISIC2016,
                        trainids=trainids, 
                        testids=testids,
                        validationids=validationids,
                        trainlabels=trainlabels_binary_ISIC2016,
                        testlabels=testlabels_binary_ISIC2016,
                        validationlabels=validationlabels_binary_ISIC2016
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename, 
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016], 
            train_only=False,
            val_exists=False, 
            test_exists=True)




        # Augmentation only on training set


        if augment_ratio is not None and augment_ratio >= 1.0:


            df_mel_augmented, df_non_mel_augmented, trainpixels_ISIC2016_augmented, \
            trainlabels_binary_ISIC2016_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_ISIC2016, 
                trainlabels=trainlabels_binary_ISIC2016,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_training_ISIC2016
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_ISIC2016_augmented, 
                            testpxs=testpixels_ISIC2016, 
                            validationpxs=validationpixels_ISIC2016,
                            trainids=trainids_new, 
                            testids=testids,
                            validationids=validationids,
                            trainlabels=trainlabels_binary_ISIC2016_augmented,
                            testlabels=testlabels_binary_ISIC2016,
                            validationlabels=validationlabels_binary_ISIC2016
                            )