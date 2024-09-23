from .parser import *


class parser_ISIC2018(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        self.base_dir = base_dir
        # ISIC2018
        self.classes_training_ISIC2018 = {
            'benign' : 'benign',
            'malignant' : 'malignant',
        }


    def saveDatasetToFile(self):
        datasetname = mel.DatasetType.ISIC2018.name

        # Set path values
        self.makeFolders(datasetname)

        ISIC2018_training_path = pathlib.Path(self.base_dir).joinpath('data', f'./{datasetname}', './ISIC2018_Task3_Training_Input')
        ISIC2018_val_path = pathlib.Path(self.base_dir).joinpath('data', f'./{datasetname}', './ISIC2018_Task3_Validation_Input')
        ISIC2018_test_path = pathlib.Path(self.base_dir).joinpath('data', f'./{datasetname}', './ISIC2018_Task3_Test_Input')

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
        imageid_path_training_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_training_path, '*.*'))}
        imageid_path_val_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_val_path, '*.*'))}
        imageid_path_test_dict_ISIC2018 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(ISIC2018_test_path, '*.*'))}

        
        # ISIC2018_columns = ['image_id', 'label']
        df_training_ISIC2018 = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
            'data', f'./{datasetname}', './ISIC2018_Task3_Training_GroundTruth', './ISIC2018_Task3_Training_GroundTruth.csv')),
            header=0)
        df_val_ISIC2018 = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
            'data', f'./{datasetname}', './ISIC2018_Task3_Validation_GroundTruth', './ISIC2018_Task3_Validation_GroundTruth.csv')),
            header=0)
        df_test_ISIC2018 = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(
            'data', f'./{datasetname}', './ISIC2018_Task3_Test_GroundTruth', './ISIC2018_Task3_Test_GroundTruth.csv')),
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
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            ))
        
        df_val_ISIC2018['image'] = df_val_ISIC2018.path.map(
            lambda x:(
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            ))

        
        df_test_ISIC2018['image'] = df_test_ISIC2018.path.map(
            lambda x:(
                img := self.encode(Image.open(x).convert("RGB")),
                currentPath := pathlib.Path(x), # [1]: PosixPath
            ))



        assert all(df_training_ISIC2018.cell_type_binary.unique() == df_test_ISIC2018.cell_type_binary.unique())
        assert all(df_val_ISIC2018.cell_type_binary.unique() == df_test_ISIC2018.cell_type_binary.unique())
        labels = df_training_ISIC2018.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)

        mel.Preprocess().saveNumpyImagesToFiles(df_training_ISIC2018, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_val_ISIC2018, self.val_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_test_ISIC2018, self.test_rgb_folder)

        # ISIC2018 binary images/labels
        trainpixels_ISIC2018 = list(map(lambda x:x[0], df_training_ISIC2018['image'])) # Filter out only pixel from the list
        validationpixels_ISIC2018 = list(map(lambda x:x[0], df_val_ISIC2018['image'])) # Filter out only pixel from the list
        testpixels_ISIC2018 = list(map(lambda x:x[0], df_test_ISIC2018['image'])) # Filter out only pixel from the list

        trainlabels_binary_ISIC2018 = np.asarray(df_training_ISIC2018['cell_type_binary_idx'], dtype='float64')
        validationlabels_binary_ISIC2018 = np.asarray(df_val_ISIC2018['cell_type_binary_idx'], dtype='float64')
        testlabels_binary_ISIC2018 = np.asarray(df_test_ISIC2018['cell_type_binary_idx'], dtype='float64')

        trainids = list(map(lambda x:x[1].stem, df_training_ISIC2018['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, df_val_ISIC2018['image']))
        testids = list(map(lambda x:x[1].stem, df_test_ISIC2018['image']))
        


        assert num_train_img_ISIC2018 == len(trainpixels_ISIC2018)
        assert num_val_img_ISIC2018 == len(validationpixels_ISIC2018)
        assert num_test_img_ISIC2018 == len(testpixels_ISIC2018)
        assert len(trainpixels_ISIC2018) == trainlabels_binary_ISIC2018.shape[0]
        assert len(validationpixels_ISIC2018) == validationlabels_binary_ISIC2018.shape[0]
        assert len(testpixels_ISIC2018) == testlabels_binary_ISIC2018.shape[0]

            

    # @staticmethod
    # def evaluate(dbpath, model_path, model_name):
    #     traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
    #     assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
    #         == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['trainimages'] + \
    #             mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['validationimages'] + \
    #                 mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['testimages']
    #     assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
    #         == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['trainimages'] + \
    #             mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['validationimages'] + \
    #                 mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['testimages']
    #     assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
    #         == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['trainimages'] + \
    #             mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['validationimages'] + \
    #                 mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2018]['testimages']

    #     testimages_decoded = []
    #     for idx, img in enumerate(testdata['testimages']):
    #             decoded_img = img_to_array(mel.Parser.decode(img))
    #             decoded_img = mel.Preprocess.normalizeImg(decoded_img)
    #             testimages_decoded.append(decoded_img)
    #     testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

    #     print('Testing on ISIC2018 DB')
    #     print(f'Evaluating {model_name} model on {mel.DatasetType.ISIC2018.name}...\n')
    #     model = load_model(model_path+'/'+model_name + '.hdf5')
    #     # model, _, _ = mel.Model.evaluate_model(
    #     #     model_name=model_name,
    #     #     model_path=model_path,
    #     #     target_db=mel.DatasetType.ISIC2018.name,
    #     #     trainimages=None,
    #     #     trainlabels=None,
    #     #     validationimages=None,
    #     #     validationlabels=None,
    #     #     testimages=testimages_decoded,
    #     #     testlabels=np.array(testdata['testlabels']),
    #     #     )
    #     target_network = model.layers[0].name

    #     test_pred, test_pred_classes = mel.Model.computing_prediction(
    #         model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2018.name, \
    #         testimages = testimages_decoded)
        
    #     test_report = mel.Model.model_report(
    #         model_name = model_name, model_path=model_path, target_db=mel.DatasetType.ISIC2018.name, \
    #             target_network = target_network, \
    #                 testlabels = np.array(testdata['testlabels']), test_pred_classes = test_pred_classes
    #     )

    #     performance = {
    #         'y_pred': test_pred_classes.tolist(),
    #         'accuracy': test_report['accuracy'],
    #         'precision': test_report['macro avg']['precision'],
    #         'sensitivity': test_report['Malignant']['recall'],
    #         'specificity': test_report['Benign']['recall'],
    #         'f1-score': test_report['macro avg']['f1-score'],
    #     }

    #     return performance