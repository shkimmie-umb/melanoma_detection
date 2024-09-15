from .parser import *


class parser_ISIC2016(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        # ISIC2016
        # self.lesion_type_binary_dict_training = {
        #     'benign' : 'Non-Melanoma',
        #     'malignant' : 'Melanoma',
        # }
        self.lesion_type_binary_dict_test = {
            0.0 : 'benign',
            1.0 : 'malignant',
        }


    def saveDatasetToFile(self):
        datasetname = mel.DatasetType.ISIC2016.name

        self.makeFolders(datasetname)

        training_path = pathlib.Path(self.base_dir).joinpath('data', datasetname, 'ISBI2016_ISIC_Part3_Training_Data')
        test_path = pathlib.Path(self.base_dir).joinpath('data', datasetname, './ISBI2016_ISIC_Part3_Test_Data')

        num_train_img = len(list(training_path.glob('./*.jpg'))) # counts all ISIC2016 training images
        num_test_img = len(list(test_path.glob('./*.jpg'))) # counts all ISIC2016 test images

        assert num_train_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['trainimages']
        assert num_test_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['testimages']

        self.logger.debug('%s %s', "Images available in ISIC2016 train dataset:", num_train_img)
        self.logger.debug('%s %s', "Images available in ISIC2016 test dataset:", num_test_img)

        # ISIC2016: Dictionary for Image Names
        imageid_path_training_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(training_path, '*.jpg'))}
        imageid_path_test_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(test_path, '*.jpg'))}
        ISIC2016_columns = ['image_id', 'label']
        df_training = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
        'data', datasetname, 'ISBI2016_ISIC_Part3_Training_GroundTruth.csv')), names=ISIC2016_columns, header=None)
        df_test = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
        'data', datasetname, 'ISBI2016_ISIC_Part3_Test_GroundTruth.csv')), names=ISIC2016_columns, header=None)

        self.logger.debug("Let's check ISIC2016 metadata briefly")
        self.logger.debug("This is ISIC2016 training data")
        display(df_training.head())
        self.logger.debug("This is ISIC2016 test data")
        display(df_test.head())


        classes_binary = df_training.label.unique() # second column is label
        num_classes_binary = len(classes_binary)
        classes_binary, num_classes_binary

        # ISIC2016: Creating New Columns for better readability
        df_training['path'] = df_training.image_id.map(imageid_path_training_dict.get)
        
        df_training['cell_type_binary'] = df_training['label']
        
        # Define codes for compatibility among datasets
        df_training['cell_type_binary_idx'] = pd.CategoricalIndex(df_training.cell_type_binary, categories=self.classes_melanoma_binary).codes
        df_test['path'] = df_test.image_id.map(imageid_path_test_dict.get)
        df_test['cell_type_binary'] = df_test['label'].map(self.lesion_type_binary_dict_test.get)
        # Define codes for compatibility among datasets
        df_test['cell_type_binary_idx'] = pd.CategoricalIndex(df_test.cell_type_binary, categories=self.classes_melanoma_binary).codes

        assert all(df_training['cell_type_binary'].isin(self.classes_melanoma_binary)) == True
        assert all(df_test['cell_type_binary'].isin(self.classes_melanoma_binary)) == True
        
        self.logger.debug("ISIC2016 training df")
        display(df_training.sample(10))
        self.logger.debug("ISIC2016 test df")
        display(df_test.sample(10))

        self.logger.debug("Check null data in ISIC2016 training metadata -> df_training.isnull().sum()")
        display(df_training.isnull().sum())
        self.logger.debug("Check null data in ISIC2016 test metadata -> df_test.isnull().sum()")
        display(df_test.isnull().sum())


        df_training['image'] = df_training.path.map(
        lambda x:(
            img := self.encode(Image.open(x).convert("RGB")),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )


        df_test['image'] = df_test.path.map(
        lambda x:(
            img := self.encode(Image.open(x).convert("RGB")),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )



        assert all(df_training.cell_type_binary.unique() == df_test.cell_type_binary.unique())
        labels = df_training.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.whole_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)


        # Dividing ISIC2016 into train/val set
        
        trainset, validationset = train_test_split(df_training, test_size=self.split_ratio, random_state = self.pseudo_num)
        # ISIC2016 test data is given, so there is no need to create test dataset separately
        # testset_ISIC2016 = df_test


        mel.Preprocess().saveNumpyImagesToFiles(trainset, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(validationset, self.val_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_test, self.test_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_training, self.whole_rgb_folder)


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
        trainpixels = list(map(lambda x:x[0], trainset['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], validationset['image']))
        testpixels = list(map(lambda x:x[0], df_test['image']))

        trainids = list(map(lambda x:x[1].stem, trainset['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, validationset['image']))
        testids = list(map(lambda x:x[1].stem, df_test['image']))

        # trainimages_ISIC2016 = preprocessor.normalizeImgs(imgs=trainpixels_ISIC2016, networktype=networktype, 
        # 										 uniform_normalization=uniform_normalization)
        # validationimages_ISIC2016 = preprocessor.normalizeImgs(imgs=validationpixels_ISIC2016, networktype=networktype,
        # 											  uniform_normalization=uniform_normalization)
        # testimages_ISIC2016 = preprocessor.normalizeImgs(imgs=testpixels_ISIC2016, networktype=networktype,
        # 										uniform_normalization=uniform_normalization)

        trainlabels_binary = np.asarray(trainset['cell_type_binary_idx'], dtype='float64')
        validationlabels_binary = np.asarray(validationset['cell_type_binary_idx'], dtype='float64')
        testlabels_binary = np.asarray(df_test['cell_type_binary_idx'], dtype='float64')

        assert num_train_img == (len(trainpixels) + len(validationpixels))
        assert num_test_img == len(testpixels)
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        assert len(testpixels) == testlabels_binary.shape[0]

        # trainimages_ISIC2016 = trainimages_ISIC2016.reshape(trainimages_ISIC2016.shape[0], *image_shape)

    @staticmethod
    def evaluate(dbpath, model_path, model_name):
        traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
        assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['testimages']
        assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['testimages']
        assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2016]['testimages']
        testimages_decoded = []
        for idx, img in enumerate(testdata['testimages']):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

        print('Testing on ISIC2016 DB')
        print(f'Evaluating {model_name} model on {mel.DatasetType.ISIC2016.name}...\n')
        model = load_model(model_path+'/'+model_name + '.hdf5')
        # model, _, _ = mel.Model.evaluate_model(
        #     model_name=model_name,
        #     model_path=model_path,
        #     target_db=mel.DatasetType.ISIC2016.name,
        #     trainimages=None,
        #     trainlabels=None,
        #     validationimages=None,
        #     validationlabels=None,
        #     testimages=testimages_decoded,
        #     testlabels=np.array(testdata['testlabels']),
        #     )
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.computing_prediction(
            model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2016.name, \
            testimages = testimages_decoded)
        
        test_report = mel.Model.model_report(
            model_name = model_name, model_path=model_path, target_db=mel.DatasetType.ISIC2016.name, \
                target_network = target_network, \
                    testlabels = np.array(testdata['testlabels']), test_pred_classes = test_pred_classes
        )

        performance = {
            'y_pred': test_pred_classes.tolist(),
            'accuracy': test_report['accuracy'],
            'precision': test_report['macro avg']['precision'],
            'sensitivity': test_report['Malignant']['recall'],
            'specificity': test_report['Benign']['recall'],
            'f1-score': test_report['macro avg']['f1-score'],
        }

        return performance