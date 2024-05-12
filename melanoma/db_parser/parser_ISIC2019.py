from .parser import *


class parser_ISIC2019(Parser):

    def __init__(self, base_dir, square_size=None, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.ISIC2019.name

        self.makeFolders(datasetname)

        ISIC2019_training_path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC_2019_Training_Input')

        num_train_img_ISIC2019 = len(list(ISIC2019_training_path.glob('./*.jpg'))) # counts all ISIC2019 training images

        assert num_train_img_ISIC2019 == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2019]['trainimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} train dataset:", num_train_img_ISIC2019)

        # ISIC2019: Dictionary for Image Names
        imageid_path_training_dict_ISIC2019 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2019_training_path, '*.jpg'))}

        
        # ISIC2018_columns = ['image_id', 'label']
        df_training_ISIC2019 = pd.read_csv(str(pathlib.Path.joinpath(
            self.base_dir, './melanomaDB', f'./{datasetname}', './ISIC_2019_Training_GroundTruth.csv')),
            header=0)

        assert df_training_ISIC2019.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2019]['trainimages']
        

        self.logger.debug("Let's check ISIC2019 metadata briefly")
        self.logger.debug("This is ISIC2019 training data samples")
        display(df_training_ISIC2019.head())



        # ISIC2019: Creating New Columns for better readability
        df_training_ISIC2019['path'] = df_training_ISIC2019['image'].map(imageid_path_training_dict_ISIC2019.get)
        df_training_ISIC2019['cell_type_binary'] = df_training_ISIC2019['MEL'].map(self.common_binary_label.get)
        df_training_ISIC2019['cell_type_binary_idx'] = pd.CategoricalIndex(df_training_ISIC2019.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in ISIC2019 training metadata")
        display(df_training_ISIC2019.isnull().sum())
        
        df_training_ISIC2019['image'] = df_training_ISIC2019.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )



        # assert all(df_training_ISIC2019.cell_type_binary.unique() == df_test_ISIC2019.cell_type_binary.unique())
        # assert all(df_val_ISIC2019.cell_type_binary.unique() == df_test_ISIC2019.cell_type_binary.unique())
        labels = df_training_ISIC2019.cell_type_binary.unique()

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

        # Dividing ISIC2019 into train/val set
        trainset_ISIC2019, validationset_ISIC2019 = train_test_split(df_training_ISIC2019, test_size=0.2, random_state = self.pseudo_num)

        self.preprocessor.saveNumpyImagesToFiles(trainset_ISIC2019, df_training_ISIC2019, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(validationset_ISIC2019, df_training_ISIC2019, self.val_rgb_folder)

        # ISIC2019 binary images/labels
        trainpixels_ISIC2019 = list(map(lambda x:x[0], trainset_ISIC2019['image'])) # Filter out only pixel from the list
        validationpixels_ISIC2019 = list(map(lambda x:x[0], validationset_ISIC2019['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, trainset_ISIC2019['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset_ISIC2019['image']))
        

        # trainimages_ISIC2019 = self.preprocessor.normalizeImgs(trainpixels_ISIC2019, networktype)
        # validationimages_ISIC2019 = self.preprocessor.normalizeImgs(validationpixels_ISIC2019, networktype)

        trainlabels_binary_ISIC2019 = to_categorical(trainset_ISIC2019.cell_type_binary_idx, num_classes=2)
        validationlabels_binary_ISIC2019 = to_categorical(validationset_ISIC2019.cell_type_binary_idx, num_classes=2)

        assert num_train_img_ISIC2019 == len(trainpixels_ISIC2019) + len(validationpixels_ISIC2019)
        assert len(trainpixels_ISIC2019) == trainlabels_binary_ISIC2019.shape[0]
        assert len(validationpixels_ISIC2019) == validationlabels_binary_ISIC2019.shape[0]
        # assert trainimages_ISIC2019.shape[0] == trainlabels_binary_ISIC2019.shape[0]
        # assert validationimages_ISIC2019.shape[0] == validationlabels_binary_ISIC2019.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_ISIC2019,
                        testpxs=[],
                        validationpxs=validationpixels_ISIC2019,
                        trainids=trainids, 
                        testids=[],
                        validationids=validationids,
                        trainlabels=trainlabels_binary_ISIC2019,
                        testlabels=[],
                        validationlabels=validationlabels_binary_ISIC2019
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2019],
            train_only=False,
            val_exists=True, 
            test_exists=False)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_ISIC2019_augmented, \
            trainlabels_binary_ISIC2019_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_ISIC2019,
                trainlabels=trainlabels_binary_ISIC2019,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_training_ISIC2019
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_ISIC2019_augmented, 
                            testpxs=[],
                            validationpxs=validationpixels_ISIC2019,
                            trainids=trainids_new, 
                            testids=[],
                            validationids=validationids,
                            trainlabels=trainlabels_binary_ISIC2019_augmented,
                            testlabels=[],
                            validationlabels=validationlabels_binary_ISIC2019
                            )