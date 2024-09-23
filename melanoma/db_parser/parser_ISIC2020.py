from .parser import *


class parser_ISIC2020(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        # ISIC2020
        # self.lesion_type_binary_dict_training_ISIC2020 = {
        #     'benign' : 'Non-Melanoma',
        #     'malignant' : 'Melanoma',
        # }


    def saveDatasetToFile(self):
        datasetname = mel.DatasetType.ISIC2020.name

        self.makeFolders(datasetname)

        training_path = pathlib.Path(self.base_dir).joinpath(datasetname, './train')
        test_path = pathlib.Path(self.base_dir).joinpath(datasetname, './ISIC_2020_Test_Input')

        num_train_img = len(list(training_path.glob('./*.jpg'))) # counts all ISIC2020 training images
        num_test_img = len(list(test_path.glob('./*.jpg')))

        assert num_train_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['trainimages']
        assert num_test_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['testimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} train dataset:", num_train_img)
        self.logger.debug('%s %s', f"Images available in {datasetname} test dataset:", num_test_img)

        # ISIC2020: Dictionary for Image Names
        imageid_path_training_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(training_path, '*.jpg'))}
        imageid_path_test_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(test_path, '*.jpg'))}

        
        
        df_training = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
            datasetname, './ISIC_2020_Training_GroundTruth.csv')), header=0)
        # df_test = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
        #     'data', datasetname, './ISIC_2020_Test_Metadata.csv')), header=0)

        assert df_training.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['trainimages']
        # assert df_test.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2020]['testimages']
        

        self.logger.debug("Let's check ISIC2020 metadata briefly")
        self.logger.debug("This is ISIC2020 training data samples")
        display(df_training.head())



        # ISIC2020: Creating New Columns for better readability
        df_training['path'] = df_training['image_name'].map(imageid_path_training_dict.get)
        df_training['cell_type_binary'] = df_training['benign_malignant']
        df_training['cell_type_binary_idx'] = pd.CategoricalIndex(df_training.cell_type_binary, categories=self.classes_melanoma_binary).codes

        # df_test['path'] = df_test['image'].map(imageid_path_test_dict.get)


        self.logger.debug("Check null data in ISIC2020 training metadata")
        display(df_training.isnull().sum())
        
        df_training['image'] = df_training.path.map(
            lambda x:(
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        # df_test['image'] = df_test.path.map(
        #     lambda x:(
        #         img := self.encode(Image.open(x).convert("RGB")),
        #         currentPath := pathlib.Path(x), # [1]: PosixPath
        #     )
        # )

        labels = df_training.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)



        # Dividing ISIC2020 into train/val set
        trainset, validationset = train_test_split(df_training, test_size=0.2,random_state = self.pseudo_num)
        

        mel.Preprocess().saveNumpyImagesToFiles(trainset, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(validationset, self.val_rgb_folder)

        # ISIC2020 binary images/labels
        trainpixels = list(map(lambda x:x[0], trainset['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], validationset['image'])) # Filter out only pixel from the list
        testpixels = list(map(lambda x:x[0], df_test['image']))

        trainids = list(map(lambda x:x[1].stem, trainset['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset['image']))
        testids = list(map(lambda x:x[1].stem, df_test['image']))
        

        trainlabels_binary = np.asarray(trainset.cell_type_binary_idx, dtype='float64')
        validationlabels_binary = np.asarray(validationset.cell_type_binary_idx, dtype='float64')
        # trainlabels_binary_ISIC2020 = to_categorical(trainset.cell_type_binary_idx, num_classes=2)
        # validationlabels_binary_ISIC2020 = to_categorical(validationset.cell_type_binary_idx, num_classes=2)

        assert num_train_img == len(trainpixels) + len(validationpixels)
        assert num_test_img == len(testpixels)
        # assert num_test_img_ISIC2020 == len(testimages_id_ISIC2020)
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        # assert trainimages_ISIC2020.shape[0] == trainlabels_binary_ISIC2020.shape[0]
        # assert validationimages_ISIC2020.shape[0] == validationlabels_binary_ISIC2020.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)