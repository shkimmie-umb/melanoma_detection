from .parser import *


class parser_PH2(Parser):

    def __init__(self, base_dir, square_size, pseudo_num = 2, split_ratio=0.2, 
                 image_resize=(None, None), networktype = None, uniform_normalization=True):
        super().__init__(base_dir = base_dir, square_size = square_size, pseudo_num = pseudo_num,
                         split_ratio = split_ratio, image_resize = image_resize, networktype = networktype,
                           uniform_normalization = uniform_normalization)
        
        


    def saveDatasetToFile(self, augment_ratio=None):
        datasetname = mel.DatasetType.PH2.name

        self.makeFolders(datasetname)

        PH2path = pathlib.Path.joinpath(self.base_dir, './melanomaDB', './PH2Dataset')

        img_path =pathlib.Path.joinpath(PH2path, './PH2 Dataset images')

        num_imgs = len(list(img_path.glob('*/*_Dermoscopic_Image/*.bmp'))) # counts all PH2 training images

        assert num_imgs == 200

        self.logger.debug('%s %s', f"Images available in {datasetname} dataset:", num_imgs)

        imageid_path_dict_PH2 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(img_path, '*/*_Dermoscopic_Image/*.bmp'))}

        
        df_PH2 = pd.read_excel(str(PH2path) + '/PH2_dataset.xlsx', header=12)

        assert df_PH2.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.PH2]['trainimages']

        self.logger.debug("Let's check PH2 metadata briefly")
        self.logger.debug("This is PH2 data samples")
        display(df_PH2.head())



        # PH2: Creating New Columns for better readability
        df_PH2['path'] = df_PH2['Image Name'].map(imageid_path_dict_PH2.get)
        df_PH2['cell_type_binary'] = np.where(df_PH2['Melanoma'] == 'X', 'Melanoma', 'Non-Melanoma')
        df_PH2['cell_type_binary_idx'] = pd.CategoricalIndex(df_PH2.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in PH2 training metadata")
        display(df_PH2.isnull().sum())
        
        df_PH2['image'] = df_PH2.path.map(
            lambda x:(
                img := self.encode(self.preprocessor.squareImgsAndResize(path=x, square_size=self.square_size,
                                                            resize_width=self.resize_width, resize_height=self.resize_height)),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df_PH2.cell_type_binary.unique()

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

        # Dividing PH2 into train/val set
        # trainset_ISIC2020, validationset_ISIC2020 = train_test_split(df_training_ISIC2020, test_size=0.2,random_state = 1)
        

        self.preprocessor.saveNumpyImagesToFiles(df_PH2, df_PH2, self.train_rgb_folder)

        # PH2 binary images/labels
        trainpixels_PH2 = list(map(lambda x:x[0], df_PH2['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, df_PH2['image'])) # Filter out only pixel from the list
        
        # trainimages_PH2 = self.preprocessor.normalizeImgs(trainpixels_PH2, networktype)
        # trainlabels_binary_ISIC2017 = np.asarray(trainset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # testlabels_binary_ISIC2017 = np.asarray(testset_ISIC2017.cell_type_binary_idx, dtype='float64')
        # validationlabels_binary_ISIC2017 = np.asarray(validationset_ISIC2017.cell_type_binary_idx, dtype='float64')
        trainlabels_binary_PH2 = to_categorical(df_PH2.cell_type_binary_idx, num_classes=2)

        assert num_imgs == len(trainpixels_PH2)
        assert len(trainpixels_PH2) == trainlabels_binary_PH2.shape[0]
        # assert trainimages_PH2.shape[0] == trainlabels_binary_PH2.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

        filename = f'{datasetname}_{self.resize_height}h_{self.resize_height}w_binary.h5' # height x width
        self.generateHDF5(path=self.path, filename=filename, 
                        trainpxs=trainpixels_PH2,
                        testpxs=[],
                        validationpxs=[],
                        trainids=trainids, 
                        testids=[],
                        validationids=[],
                        trainlabels=trainlabels_binary_PH2,
                        testlabels=[],
                        validationlabels=[]
                        )
        
        self.validate_h5(
            path=self.path,
            filename=filename,
            dbnumimgs=mel.CommonData().dbNumImgs[mel.DatasetType.PH2],
            train_only=True,
            val_exists=False, 
            test_exists=False)




        if augment_ratio is not None and augment_ratio >= 1.0:

            df_mel_augmented, df_non_mel_augmented, trainpixels_PH2_augmented, \
            trainlabels_binary_PH2_augmented, trainids_augmented = \
            self.preprocessor.augmentation(
                train_rgb_folder=self.train_rgb_folder, 
                labels=labels, 
                trainimages=trainpixels_PH2,
                trainlabels=trainlabels_binary_PH2,
                square_size = self.square_size, 
                resize_width = self.resize_width, 
                resize_height = self.resize_height, 
                augment_ratio = augment_ratio, 
                df_trainset = df_PH2
            )

            trainids_new = trainids + trainids_augmented


            filename_aug = f'{datasetname}_augmentedWith_{df_mel_augmented.shape[0]}Melanoma_{df_non_mel_augmented.shape[0]}Non-Melanoma_{self.resize_height}h_{self.resize_width}w_binary.h5'


            # create HDF5 file
            self.generateHDF5(path=self.path, filename=filename_aug, 
                            trainpxs=trainpixels_PH2_augmented, 
                            testpxs=[],
                            validationpxs=[],
                            trainids=trainids_new, 
                            testids=[],
                            validationids=[],
                            trainlabels=trainlabels_binary_PH2_augmented,
                            testlabels=[],
                            validationlabels=[]
                            )