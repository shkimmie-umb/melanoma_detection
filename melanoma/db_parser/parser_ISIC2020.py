from .parser import *


class parser_ISIC2020(Parser):

    def __init__(self, base_dir, square_size, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        # ISIC2020
        self.lesion_type_binary_dict_training_ISIC2020 = {
            'benign' : 'Non-Melanoma',
            'malignant' : 'Melanoma',
        }


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.ISIC2020.name

        self.makeFolders(datasetname)

        ISIC2020_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasetname}', './train')
        ISIC2020_test_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC_2020_Test_Input')

        num_train_img_ISIC2020 = len(list(ISIC2020_training_path.glob('./*.jpg'))) # counts all ISIC2020 training images
        num_test_img_ISIC2020 = len(list(ISIC2020_test_path.glob('./*.jpg')))

        assert num_train_img_ISIC2020 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['trainimages']
        assert num_test_img_ISIC2020 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['testimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} train dataset:", num_train_img_ISIC2020)
        self.logger.debug('%s %s', f"Images available in {datasetname} test dataset:", num_test_img_ISIC2020)

        # ISIC2020: Dictionary for Image Names
        imageid_path_training_dict_ISIC2020 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2020_training_path, '*.jpg'))}
        imageid_path_test_dict_ISIC2020 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2020_test_path, '*.jpg'))}

        
        
        df_training_ISIC2020 = pd.read_csv(str(pathlib.Path.joinpath(
            self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC_2020_Training_GroundTruth.csv')),
            header=0)
        df_test_ISIC2020 = pd.read_csv(str(pathlib.Path.joinpath(
            self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC_2020_Test_Metadata.csv')),
            header=0)

        assert df_training_ISIC2020.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['trainimages']
        assert df_test_ISIC2020.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['testimages']
        

        self.logger.debug("Let's check ISIC2020 metadata briefly")
        self.logger.debug("This is ISIC2020 training data samples")
        display(df_training_ISIC2020.head())



        # ISIC2020: Creating New Columns for better readability
        df_training_ISIC2020['path'] = df_training_ISIC2020['image_name'].map(imageid_path_training_dict_ISIC2020.get)
        df_training_ISIC2020['cell_type_binary'] = df_training_ISIC2020['benign_malignant'].map(self.lesion_type_binary_dict_training_ISIC2020.get)
        df_training_ISIC2020['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2020.cell_type_binary, categories=self.classes_melanoma_binary).codes

        df_test_ISIC2020['path'] = df_test_ISIC2020['image'].map(imageid_path_test_dict_ISIC2020.get)


        self.logger.debug("Check null data in ISIC2020 training metadata")
        display(df_training_ISIC2020.isnull().sum())
        
        df_training_ISIC2020['image'] = df_training_ISIC2020.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        df_test_ISIC2020['image'] = df_test_ISIC2020.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )



        labels = df_training_ISIC2020.cell_type_binary.unique()

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



        # Dividing ISIC2020 into train/val set
        trainset_ISIC2020, validationset_ISIC2020 = train_test_split(df_training_ISIC2020, test_size=0.2,random_state = self.pseudo_num)
        testset_ISIC2020 = df_test_ISIC2020

        self.preprocessor.saveNumpyImagesToFiles(trainset_ISIC2020, df_training_ISIC2020, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(validationset_ISIC2020, df_training_ISIC2020, self.val_rgb_folder)
        # self.preprocessor.saveNumpyImagesToFilesWithoutLabel(testset_ISIC2020, self.test_rgb_folder)

        # ISIC2020 binary images/labels
        trainpixels_ISIC2020 = list(map(lambda x:x[0], trainset_ISIC2020['image'])) # Filter out only pixel from the list
        validationpixels_ISIC2020 = list(map(lambda x:x[0], validationset_ISIC2020['image'])) # Filter out only pixel from the list
        testpixels_ISIC2020 = list(map(lambda x:x[0], testset_ISIC2020['image']))
        # testimages_id_ISIC2020 = list(map(lambda x:x[2], testset_ISIC2020['image']))

        trainids = list(map(lambda x:x[1].stem, trainset_ISIC2020['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset_ISIC2020['image']))
        

        # trainimages_ISIC2020 = preprocessor.normalizeImgs(trainpixels_ISIC2020, networktype)
        # validationimages_ISIC2020 = preprocessor.normalizeImgs(validationpixels_ISIC2020, networktype)
        # testimages_ISIC2020 = preprocessor.normalizeImgs(testpixels_ISIC2020, networktype)
        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary_ISIC2020 = to_categorical(trainset_ISIC2020.cell_type_binary_idx, num_classes=2)
        validationlabels_binary_ISIC2020 = to_categorical(validationset_ISIC2020.cell_type_binary_idx, num_classes=2)

        # assert num_train_img_ISIC2020 == len(trainpixels_ISIC2020) + len(validationpixels_ISIC2020)
        assert num_test_img_ISIC2020 == len(testpixels_ISIC2020)
        # assert num_test_img_ISIC2020 == len(testimages_id_ISIC2020)
        assert len(trainpixels_ISIC2020) == trainlabels_binary_ISIC2020.shape[0]
        assert len(validationpixels_ISIC2020) == validationlabels_binary_ISIC2020.shape[0]
        # assert trainimages_ISIC2020.shape[0] == trainlabels_binary_ISIC2020.shape[0]
        # assert validationimages_ISIC2020.shape[0] == validationlabels_binary_ISIC2020.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_ISIC2020,
                        testpxs=[],
                        validationpxs=validationpixels_ISIC2020,
                        trainids=trainids, 
                        testids=[],
                        validationids=validationids,
                        trainlabels=trainlabels_binary_ISIC2020,
                        testlabels=[],
                        validationlabels=validationlabels_binary_ISIC2020
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020],
            train_only=False,
            val_exists=True, 
            test_exists=False)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_ISIC2020_augmented, \
            trainlabels_binary_ISIC2020_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_ISIC2020,
                trainlabels=trainlabels_binary_ISIC2020,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_training_ISIC2020
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_ISIC2020_augmented, 
                            testpxs=[],
                            validationpxs=validationpixels_ISIC2020,
                            trainids=trainids_new, 
                            testids=[],
                            validationids=validationids,
                            trainlabels=trainlabels_binary_ISIC2020_augmented,
                            testlabels=[],
                            validationlabels=validationlabels_binary_ISIC2020
                            )