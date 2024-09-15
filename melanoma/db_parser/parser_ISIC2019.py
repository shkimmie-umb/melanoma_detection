from .parser import *


class parser_ISIC2019(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        self.base_dir = base_dir

    def saveDatasetToFile(self):
        datasetname = mel.DatasetType.ISIC2019.name

        self.makeFolders(datasetname)

        training_path = pathlib.Path(self.base_dir).joinpath('data', f'./{datasetname}', 'ISIC_2019_Training_Input')

        num_train_img = len(list(training_path.glob('./*.jpg'))) # counts all ISIC2019 training images

        assert num_train_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2019]['trainimages']

        self.logger.debug('%s %s', f"Images available in {datasetname} train dataset:", num_train_img)

        # ISIC2019: Dictionary for Image Names
        imageid_path_training_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(training_path, '*.*'))}

        
        # ISIC2018_columns = ['image_id', 'label']
        df_training = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
            'data', f'./{datasetname}', 'ISIC_2019_Training_GroundTruth.csv')),
            header=0)

        assert df_training.shape[0] == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2019]['trainimages']
        

        self.logger.debug("Let's check ISIC2019 metadata briefly")
        self.logger.debug("This is ISIC2019 training data samples")
        display(df_training.head())



        # ISIC2019: Creating New Columns for better readability
        df_training['path'] = df_training['image'].map(imageid_path_training_dict.get)
        df_training['cell_type_binary'] = df_training['MEL'].map(self.common_binary_label.get)
        df_training['cell_type_binary_idx'] = pd.CategoricalIndex(df_training.cell_type_binary, categories=self.classes_melanoma_binary).codes


        self.logger.debug("Check null data in ISIC2019 training metadata")
        display(df_training.isnull().sum())
        
        df_training['image'] = df_training.path.map(
            lambda x:(
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        labels = df_training.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)

        # Dividing ISIC2019 into train/val set
        trainset, validationset = train_test_split(df_training, test_size=0.2, random_state = self.pseudo_num)

        mel.Preprocess().saveNumpyImagesToFiles(trainset, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(validationset, self.val_rgb_folder)

        # ISIC2019 binary images/labels
        trainpixels = list(map(lambda x:x[0], trainset['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], validationset['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, trainset['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset['image']))

        # trainlabels_binary_ISIC2019 = to_categorical(trainset_ISIC2019.cell_type_binary_idx, num_classes=2)
        # validationlabels_binary_ISIC2019 = to_categorical(validationset_ISIC2019.cell_type_binary_idx, num_classes=2)
        trainlabels_binary = np.asarray(trainset['cell_type_binary_idx'], dtype='float64')
        validationlabels_binary = np.asarray(validationset['cell_type_binary_idx'], dtype='float64')

        assert num_train_img == len(trainpixels) + len(validationpixels)
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]

        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)