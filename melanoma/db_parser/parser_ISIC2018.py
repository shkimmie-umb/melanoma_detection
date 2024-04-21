from .parser import *


class parser_ISIC2018(Parser):

    def __init__(self, base_dir, square_size, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.ISIC2018.name

        self.makeFolders(datasetname)

        ISIC2018_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC2018_Task3_Training_Input')
        ISIC2018_val_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC2018_Task3_Validation_Input')
        ISIC2018_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC2018_Task3_Test_Input')

        num_train_img_ISIC2018 = len(list(ISIC2018_training_path.glob('./*.jpg'))) # counts all ISIC2018 training images
        num_val_img_ISIC2018 = len(list(ISIC2018_val_path.glob('./*.jpg'))) # counts all ISIC2018 validation images
        num_test_img_ISIC2018 = len(list(ISIC2018_test_path.glob('./*.jpg'))) # counts all ISIC2018 test images

        assert num_train_img_ISIC2018 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['trainimages']
        assert num_val_img_ISIC2018 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['validationimages']
        assert num_test_img_ISIC2018 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['testimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} train dataset:", num_train_img_ISIC2018)
        self.logger.debug('%s %s', f"Images available in {datasetname} validation dataset:", num_val_img_ISIC2018)
        self.logger.debug('%s %s', f"Images available in {datasetname} test dataset:", num_test_img_ISIC2018)

        # ISIC2018: Dictionary for Image Names
        imageid_path_training_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_training_path, '*.jpg'))}
        imageid_path_val_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_val_path, '*.jpg'))}
        imageid_path_test_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_test_path, '*.jpg'))}

        
        # ISIC2018_columns = ['image_id', 'label']
        df_training_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(
            self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC2018_Task3_Training_GroundTruth', './ISIC2018_Task3_Training_GroundTruth.csv')),
            header=0)
        df_val_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(
            self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC2018_Task3_Validation_GroundTruth', './ISIC2018_Task3_Validation_GroundTruth.csv')),
            header=0)
        df_test_ISIC2018 = pd.read_csv(str(pathlib.Path.joinpath(
            self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC2018_Task3_Test_GroundTruth', './ISIC2018_Task3_Test_GroundTruth.csv')),
            header=0)

        assert df_training_ISIC2018.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['trainimages']
        assert df_val_ISIC2018.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['validationimages']
        assert df_test_ISIC2018.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['testimages']

        self.logger.debug("Let's check ISIC2018 metadata briefly")
        self.logger.debug("This is ISIC2018 training data samples")
        display(df_training_ISIC2018.head())
        self.logger.debug("This is ISIC2018 validation data samples")
        display(df_val_ISIC2018.head())
        self.logger.debug("This is ISIC2018 test data samples")
        display(df_test_ISIC2018.head())



        # ISIC2018: Creating New Columns for better readability
        df_training_ISIC2018['path'] = df_training_ISIC2018['image'].map(imageid_path_training_dict_ISIC2018.get)
        df_training_ISIC2018['cell_type_binary'] = df_training_ISIC2018['MEL'].map(self.common_binary_label.get)
        df_training_ISIC2018['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2018.cell_type_binary, categories=self.classes_melanoma_binary).codes

        df_val_ISIC2018['path'] = df_val_ISIC2018['image'].map(imageid_path_val_dict_ISIC2018.get)
        df_val_ISIC2018['cell_type_binary'] = df_val_ISIC2018['MEL'].map(self.common_binary_label.get)
        df_val_ISIC2018['cell_type_binary_idx'] = pd.CategoricalIndex(df_val_ISIC2018.cell_type_binary, categories=self.classes_melanoma_binary).codes

        df_test_ISIC2018['path'] = df_test_ISIC2018['image'].map(imageid_path_test_dict_ISIC2018.get)
        df_test_ISIC2018['cell_type_binary'] = df_test_ISIC2018['MEL'].map(self.common_binary_label.get)
        df_test_ISIC2018['cell_type_binary_idx'] = pd.CategoricalIndex(df_test_ISIC2018.cell_type_binary, categories=self.classes_melanoma_binary).codes



        self.logger.debug("Check null data in ISIC2018 training metadata")
        display(df_training_ISIC2018.isnull().sum())
        self.logger.debug("Check null data in ISIC2018 validation metadata")
        display(df_val_ISIC2018.isnull().sum())
        self.logger.debug("Check null data in ISIC2018 test metadata")
        display(df_test_ISIC2018.isnull().sum())
        
        df_training_ISIC2018['image'] = df_training_ISIC2018.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )
        
        df_val_ISIC2018['image'] = df_val_ISIC2018.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        
        df_test_ISIC2018['image'] = df_test_ISIC2018.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )



        assert all(df_training_ISIC2018.cell_type_binary.unique() == df_test_ISIC2018.cell_type_binary.unique())
        assert all(df_val_ISIC2018.cell_type_binary.unique() == df_test_ISIC2018.cell_type_binary.unique())
        labels = df_training_ISIC2018.cell_type_binary.unique()

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


        # df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

        # ISIC2018 datasets are divided into train/val/test already
        trainset_ISIC2018 = df_training_ISIC2018
        validationset_ISIC2018 = df_val_ISIC2018
        testset_ISIC2018 = df_test_ISIC2018

        self.preprocessor.saveNumpyImagesToFiles(trainset_ISIC2018, df_training_ISIC2018, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(validationset_ISIC2018, df_val_ISIC2018, self.val_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(testset_ISIC2018, df_test_ISIC2018, self.test_rgb_folder)

        # ISIC2018 binary images/labels
        trainpixels_ISIC2018 = list(map(lambda x:x[0], trainset_ISIC2018['image'])) # Filter out only pixel from the list
        validationpixels_ISIC2018 = list(map(lambda x:x[0], validationset_ISIC2018['image'])) # Filter out only pixel from the list
        testpixels_ISIC2018 = list(map(lambda x:x[0], testset_ISIC2018['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, trainset_ISIC2018['image'])) # Filter out only pixel from the list
        testids = list(map(lambda x:x[1].stem, testset_ISIC2018['image']))
        validationids = list(map(lambda x:x[1].stem, validationset_ISIC2018['image']))
        
        # trainimages_ISIC2018 = preprocessor.normalizeImgs(trainpixels_ISIC2018, networktype)
        # validationimages_ISIC2018 = preprocessor.normalizeImgs(validationpixels_ISIC2018, networktype)
        # testimages_ISIC2018 = preprocessor.normalizeImgs(testpixels_ISIC2018, networktype)
        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary_ISIC2018 = to_categorical(trainset_ISIC2018.cell_type_binary_idx, num_classes=2)
        testlabels_binary_ISIC2018 = to_categorical(testset_ISIC2018.cell_type_binary_idx, num_classes=2)
        validationlabels_binary_ISIC2018 = to_categorical(validationset_ISIC2018.cell_type_binary_idx, num_classes=2)

        assert num_train_img_ISIC2018 == len(trainpixels_ISIC2018)
        assert num_val_img_ISIC2018 == len(validationpixels_ISIC2018)
        assert num_test_img_ISIC2018 == len(testpixels_ISIC2018)
        assert len(trainpixels_ISIC2018) == trainlabels_binary_ISIC2018.shape[0]
        assert len(validationpixels_ISIC2018) == validationlabels_binary_ISIC2018.shape[0]
        assert len(testpixels_ISIC2018) == testlabels_binary_ISIC2018.shape[0]
        # assert trainimages_ISIC2018.shape[0] == trainlabels_binary_ISIC2018.shape[0]
        # assert validationimages_ISIC2018.shape[0] == validationlabels_binary_ISIC2018.shape[0]
        # assert testimages_ISIC2018.shape[0] == testlabels_binary_ISIC2018.shape[0]
        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_ISIC2018,
                        testpxs=testpixels_ISIC2018,
                        validationpxs=validationpixels_ISIC2018,
                        trainids=trainids, 
                        testids=testids,
                        validationids=validationids,
                        trainlabels=trainlabels_binary_ISIC2018,
                        testlabels=testlabels_binary_ISIC2018,
                        validationlabels=validationlabels_binary_ISIC2018
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018],
            train_only=False,
            val_exists=True, 
            test_exists=True)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_ISIC2018_augmented, \
            trainlabels_binary_ISIC2018_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_ISIC2018,
                trainlabels=trainlabels_binary_ISIC2018,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_training_ISIC2018
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_ISIC2018_augmented, 
                            testpxs=testpixels_ISIC2018,
                            validationpxs=validationpixels_ISIC2018,
                            trainids=trainids_new, 
                            testids=testids,
                            validationids=validationids,
                            trainlabels=trainlabels_binary_ISIC2018_augmented,
                            testlabels=testlabels_binary_ISIC2018,
                            validationlabels=validationlabels_binary_ISIC2018
                            )
