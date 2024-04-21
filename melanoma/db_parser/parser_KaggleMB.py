from .parser import *


class parser_KaggleMB(Parser):

    def __init__(self, base_dir, square_size, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.KaggleMB.name

        self.makeFolders(datasetname)

        dbpath = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './Kaggle_malignant_benign_DB')

        num_imgs = len(list(dbpath.glob('t*/*/*.*'))) # counts all Kaggle Malignant Benign training images

        # train: 1440 benign, 1197 malignant; test: 360 benign + 300 malignant
        assert num_imgs == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['trainimages'] + mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['testimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)

        # imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(dbpath, 't*/*/*.*'))}
        paths = glob(os.path.join(dbpath, 't*/*/*.*'))
        # labels_dict = {os.path.basename(x): x for x in os.path.abspath(os.path.join(os.path.join(imageid_path_dict.values()), os.pardir))}
        df = pd.DataFrame()


        # Kaggle MB: Creating New Columns for better readability
        df['path'] = paths
        df['label'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir))))
        df['portion'] = df['path'].map(lambda x: os.path.basename(os.path.abspath(os.path.join(x, os.pardir, os.pardir))))
        assert df['label'].unique().shape[0] == 2
        df['cell_type_binary'] = np.where(df['label'] == 'malignant', 'Melanoma', 'Non-Melanoma')
        df['cell_type_binary_idx'] = pd.CategoricalIndex(df.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in Kaggle MB training metadata")
        display(df.isnull().sum())
        
        df['image'] = df.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df.cell_type_binary.unique()

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

        # Dividing Kaggle MB into train/test set
        df_trainset_temp = df.query('portion == "train"')
        df_testset = df.query('portion == "test"')

        # Dividing PAD UFES 20 into train/val set
        df_trainset, df_validationset = train_test_split(df_trainset_temp, test_size=0.2,random_state = self.pseudo_num)
        

        self.preprocessor.saveNumpyImagesToFiles(df_trainset, df, self.train_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(df_validationset, df, self.val_rgb_folder)
        self.preprocessor.saveNumpyImagesToFiles(df_testset, df, self.test_rgb_folder)

        # KaggleMB binary images/labels
        trainpixels = list(map(lambda x:x[0], df_trainset['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], df_validationset['image'])) # Filter out only pixel from the list
        testpixels = list(map(lambda x:x[0], df_testset['image'])) # Filter out only pixel from the list
        
        trainids = list(map(lambda x:x[1].stem, df_trainset['image'])) # Filter out only pixel from the list
        testids = list(map(lambda x:x[1].stem, df_testset['image']))
        validationids = list(map(lambda x:x[1].stem, df_validationset['image']))
        
        
        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary = to_categorical(df_trainset.cell_type_binary_idx, num_classes=2)
        validationlabels_binary = to_categorical(df_validationset.cell_type_binary_idx, num_classes=2)
        testlabels_binary = to_categorical(df_testset.cell_type_binary_idx, num_classes=2)

        assert len(trainpixels)+len(validationpixels) == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['trainimages']
        assert len(testpixels) == mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB]['testimages']
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        assert len(testpixels) == testlabels_binary.shape[0]
        # assert trainimages.shape[0] == trainlabels_binary.shape[0]
        # assert validationimages.shape[0] == validationlabels_binary.shape[0]
        # assert testimages.shape[0] == testlabels_binary.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels,
                        testpxs=testpixels,
                        validationpxs=validationpixels,
                        trainids=trainids, 
                        testids=testids,
                        validationids=validationids,
                        trainlabels=trainlabels_binary,
                        testlabels=testlabels_binary,
                        validationlabels=validationlabels_binary
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.KaggleMB],
            train_only=False,
            val_exists=True, 
            test_exists=True)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_KaggleMB_augmented, \
            trainlabels_binary_KaggleMB_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels,
                trainlabels=trainlabels_binary,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_trainset
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_KaggleMB_augmented, 
                            testpxs=testpixels,
                            validationpxs=validationpixels,
                            trainids=trainids_new, 
                            testids=testids,
                            validationids=validationids,
                            trainlabels=trainlabels_binary_KaggleMB_augmented,
                            testlabels=testlabels_binary,
                            validationlabels=validationlabels_binary
                            )