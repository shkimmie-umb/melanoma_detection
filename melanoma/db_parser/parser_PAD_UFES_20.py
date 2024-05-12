from .parser import *


class parser_PAD_UFES_20(Parser):

    def __init__(self, base_dir, square_size=None, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.PAD_UFES_20.name

        self.makeFolders(datasetname)

        dbpath = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './PAD-UFES-20')

        img_path =pathlib.Path.joinpath(dbpath, './images')

        num_imgs = len(list(img_path.glob('imgs_part_*/*.*'))) # counts all PAD_UFES_20 training images

        assert num_imgs == mel.CommonData().dbNumImgs[mel.DatasetType.PAD_UFES_20]['trainimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)

        imageid_path_dict = {os.path.basename(x): x for x in glob(os.path.join(img_path, 'imgs_part_*/*.*'))}

        
        df_PAD_UFES_20 = pd.read_csv(str(dbpath) + '/metadata.csv', header=0)

        assert df_PAD_UFES_20.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.PAD_UFES_20]['trainimages']

        self.logger.debug("Let's check PAD UFES 20 metadata briefly")
        self.logger.debug("This is PAD UFES 20 data samples")
        display(df_PAD_UFES_20.head())



        # PAD UFES 20: Creating New Columns for better readability
        df_PAD_UFES_20['path'] = df_PAD_UFES_20['img_id'].map(imageid_path_dict.get)
        df_PAD_UFES_20['cell_type_binary'] = np.where(df_PAD_UFES_20['diagnostic'] == 'MEL', 'Melanoma', 'Non-Melanoma')
        df_PAD_UFES_20['cell_type_binary_idx'] = pd.CategoricalIndex(df_PAD_UFES_20.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in PAD UFES 20 training metadata")
        display(df_PAD_UFES_20.isnull().sum())
        
        df_PAD_UFES_20['image'] = df_PAD_UFES_20.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )


        


        

        labels = df_PAD_UFES_20.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                
        if not self.isWholeFeatureExist or not self.isTrainFeatureExist or not self.isValFeatureExist or not self.isTestFeatureExist:
            for i in labels:
                os.makedirs(f"{self.whole_feature_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_feature_folder}/{i}", exist_ok=True)


        # df_training_ISIC2017['image'] = df_training_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_val_ISIC2017['image'] = df_val_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))
        # df_test_ISIC2017['image'] = df_test_ISIC2017.path.map(lambda x: np.asarray(Image.open(x).resize((img_width, img_height))))

        # Dividing PAD UFES 20 into train/val set
        trainset_PAD_UFES_20, validationset_PAD_UFES_20 = train_test_split(df_PAD_UFES_20, test_size=0.2,random_state = self.pseudo_num)
        

        self.preprocessor.saveNumpyImagesToFiles(df_PAD_UFES_20, df_PAD_UFES_20, self.train_rgb_folder)

        # PAD UFES 20 binary images/labels
        trainpixels_PAD_UFES_20 = list(map(lambda x:x[0], trainset_PAD_UFES_20.image)) # Filter out only pixel from the list
        validationpixels_PAD_UFES_20 = list(map(lambda x:x[0], validationset_PAD_UFES_20.image)) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, trainset_PAD_UFES_20['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset_PAD_UFES_20['image']))


        
        # trainimages_PAD_UFES_20 = preprocessor.normalizeImgs(trainpixels_PAD_UFES_20, networktype)
        # validationimages_PAD_UFES_20 = preprocessor.normalizeImgs(validationpixels_PAD_UFES_20, networktype)
        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary_PAD_UFES_20 = to_categorical(trainset_PAD_UFES_20.cell_type_binary_idx, num_classes=2)
        validationlabels_binary_PAD_UFES_20 = to_categorical(validationset_PAD_UFES_20.cell_type_binary_idx, num_classes=2)

        assert num_imgs == len(trainpixels_PAD_UFES_20) + len(validationpixels_PAD_UFES_20)
        assert len(trainpixels_PAD_UFES_20) == trainlabels_binary_PAD_UFES_20.shape[0]
        assert len(validationpixels_PAD_UFES_20) == validationlabels_binary_PAD_UFES_20.shape[0]
        # assert trainimages_PAD_UFES_20.shape[0] == trainlabels_binary_PAD_UFES_20.shape[0]
        # assert validationimages_PAD_UFES_20.shape[0] == validationlabels_binary_PAD_UFES_20.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_PAD_UFES_20,
                        testpxs=[],
                        validationpxs=validationpixels_PAD_UFES_20,
                        trainids=trainids, 
                        testids=[],
                        validationids=validationids,
                        trainlabels=trainlabels_binary_PAD_UFES_20,
                        testlabels=[],
                        validationlabels=validationlabels_binary_PAD_UFES_20
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.PAD_UFES_20],
            train_only=False,
            val_exists=True, 
            test_exists=False)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_PADUFES20_augmented, \
            trainlabels_binary_PADUFES20_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_PAD_UFES_20,
                trainlabels=trainlabels_binary_PAD_UFES_20,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = trainset_PAD_UFES_20
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_PADUFES20_augmented, 
                            testpxs=[],
                            validationpxs=validationpixels_PAD_UFES_20,
                            trainids=trainids_new, 
                            testids=[],
                            validationids=validationids,
                            trainlabels=trainlabels_binary_PADUFES20_augmented,
                            testlabels=[],
                            validationlabels=validationlabels_binary_PAD_UFES_20
                            )