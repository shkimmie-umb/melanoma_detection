from .parser import *


class parser_ISIC2017(Parser):

    def __init__(self, base_dir, pseudo_num = 2, split_ratio=0.2):
        super().__init__(base_dir = base_dir, pseudo_num = pseudo_num, split_ratio = split_ratio)
        
        # ISIC2017
        self.lesion_type_dict_ISIC2017_task3_1 = { # Official ISIC2017 task 3 - 1
            0.0: 'nevus or seborrheic keratosis',
            1.0: 'melanoma'
        }
        self.lesion_type_dict_ISIC2017_task3_2 = { # Official ISIC2017 task 3 - 2
            0.0: 'melanoma or nevus',
            1.0: 'seborrheic keratosis',
        }
        self.lesion_type_binary_dict_ISIC2017 = { # Binary melanoma detection
            0.0: 'benign',
            1.0: 'malignant',
        }

        self.classes_ISIC2017_task3_1 = ['nevus or seborrheic keratosis', 'melanoma']
        self.classes_ISIC2017_task3_2 = ['melanoma or nevus', 'seborrheic keratosis']


    def saveDatasetToFile(self):
        datasetname = mel.DatasetType.ISIC2017.name

        self.makeFolders(datasetname)

        training_path = pathlib.Path(self.base_dir).joinpath(datasetname, 'ISIC-2017_Training_Data')
        val_path = pathlib.Path(self.base_dir).joinpath(datasetname, 'ISIC-2017_Validation_Data')
        test_path = pathlib.Path(self.base_dir).joinpath(datasetname, 'ISIC-2017_Test_v2_Data')

        num_train_img = len(list(training_path.glob('./*.jpg'))) # counts all ISIC2017 training images
        num_val_img = len(list(val_path.glob('./*.jpg'))) # counts all ISIC2017 validation images
        num_test_img = len(list(test_path.glob('./*.jpg'))) # counts all ISIC2017 test images

        assert num_train_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages']
        assert num_val_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages']
        assert num_test_img == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']

        self.logger.debug('%s %s', "Images available in ISIC2017 train dataset:", num_train_img)
        self.logger.debug('%s %s', "Images available in ISIC2017 validation dataset:", num_val_img)
        self.logger.debug('%s %s', "Images available in ISIC2017 test dataset:", num_test_img)

        # ISIC2017: Dictionary for Image Names
        imageid_path_training_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(training_path, '*.jpg'))}
        imageid_path_val_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(val_path, '*.jpg'))}
        imageid_path_test_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(test_path, '*.jpg'))}


        df_training = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(datasetname, 'ISIC-2017_Training_Part3_GroundTruth.csv')))
        df_val = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(datasetname, 'ISIC-2017_Validation_Part3_GroundTruth.csv')))
        df_test = pd.read_csv(str(pathlib.Path(self.base_dir).joinpath(datasetname, 'ISIC-2017_Test_v2_Part3_GroundTruth.csv')))


        self.logger.debug("Let's check ISIC2017 metadata briefly")
        self.logger.debug("This is ISIC2017 training data samples")
        # No need to create column titles (1st row) as ISIC2017 has default column titles
        display(df_training.head())
        self.logger.debug("This is ISIC2017 validation data samples")
        display(df_val.head())
        self.logger.debug("This is ISIC2017 test data samples")
        display(df_test.head())

        

        # ISIC2017: Creating New Columns for better readability
        df_training['path'] = df_training.image_id.map(imageid_path_training_dict.get)
        df_training['cell_type_binary'] = df_training.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
        df_training['cell_type_task3_1'] = df_training.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
        df_training['cell_type_task3_2'] = df_training.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
        df_training['cell_type_binary_idx'] = pd.CategoricalIndex(df_training.cell_type_binary, categories=self.classes_melanoma_binary).codes
        df_training['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_training.cell_type_task3_1, categories=self.classes_ISIC2017_task3_1).codes
        df_training['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_training.cell_type_task3_2, categories=self.classes_ISIC2017_task3_2).codes

        df_val['path'] = df_val.image_id.map(imageid_path_val_dict.get)
        df_val['cell_type_binary'] = df_val.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
        df_val['cell_type_task3_1'] = df_val.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
        df_val['cell_type_task3_2'] = df_val.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
        df_val['cell_type_binary_idx'] = pd.CategoricalIndex(df_val.cell_type_binary, categories=self.classes_melanoma_binary).codes
        df_val['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_val.cell_type_task3_1, categories=self.classes_ISIC2017_task3_1).codes
        df_val['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_val.cell_type_task3_2, categories=self.classes_ISIC2017_task3_2).codes

        df_test['path'] = df_test.image_id.map(imageid_path_test_dict.get)
        df_test['cell_type_binary'] = df_test.melanoma.map(self.lesion_type_binary_dict_ISIC2017.get)
        df_test['cell_type_task3_1'] = df_test.melanoma.map(self.lesion_type_dict_ISIC2017_task3_1.get)
        df_test['cell_type_task3_2'] = df_test.melanoma.map(self.lesion_type_dict_ISIC2017_task3_2.get)
        df_test['cell_type_binary_idx'] = pd.CategoricalIndex(df_test.cell_type_binary, categories=self.classes_melanoma_binary).codes
        df_test['cell_type_task3_1_idx'] = pd.CategoricalIndex(df_test.cell_type_task3_1, categories=self.classes_ISIC2017_task3_1).codes
        df_test['cell_type_task3_2_idx'] = pd.CategoricalIndex(df_test.cell_type_task3_2, categories=self.classes_ISIC2017_task3_2).codes



        self.logger.debug("Check null data in ISIC2017 training metadata")
        display(df_training.isnull().sum())
        self.logger.debug("Check null data in ISIC2017 validation metadata")
        display(df_val.isnull().sum())
        self.logger.debug("Check null data in ISIC2017 test metadata")
        display(df_test.isnull().sum())

        df_training['image'] = df_training.path.map(
        lambda x:(
            img := self.encode(Image.open(x).convert("RGB")),
            currentPath := pathlib.Path(x), # [1]: PosixPath
            )
        )

        df_val['image'] = df_val.path.map(
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
        assert all(df_val.cell_type_binary.unique() == df_test.cell_type_binary.unique())
        labels = df_training.cell_type_binary.unique()

        if not self.isWholeRGBExist or not self.isTrainRGBExist or not self.isValRGBExist or not self.isTestRGBExist:
            for i in labels:
                os.makedirs(f"{self.train_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.val_rgb_folder}/{i}", exist_ok=True)
                os.makedirs(f"{self.test_rgb_folder}/{i}", exist_ok=True)

        mel.Preprocess().saveNumpyImagesToFiles(df_training, self.train_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_val, self.val_rgb_folder)
        mel.Preprocess().saveNumpyImagesToFiles(df_test, self.test_rgb_folder)

        # ISIC2017 binary images/labels
        trainpixels = list(map(lambda x:x[0], df_training['image'])) # Filter out only pixel from the list
        validationpixels = list(map(lambda x:x[0], df_val['image'])) # Filter out only pixel from the list
        testpixels = list(map(lambda x:x[0], df_test['image'])) # Filter out only pixel from the list

        trainids = list(map(lambda x:x[1].stem, df_training['image'])) # Filter out only pixel from the list
        validationids = list(map(lambda x:x[1].stem, df_val['image']))
        testids = list(map(lambda x:x[1].stem, df_test['image']))

        trainlabels_binary = np.asarray(df_training.cell_type_binary_idx, dtype='float64')
        validationlabels_binary = np.asarray(df_val.cell_type_binary_idx, dtype='float64')
        testlabels_binary = np.asarray(df_test.cell_type_binary_idx, dtype='float64')

        assert num_train_img == len(trainpixels)
        assert num_val_img == len(validationpixels)
        assert num_test_img == len(testpixels)
        assert len(trainpixels) == trainlabels_binary.shape[0]
        assert len(validationpixels) == validationlabels_binary.shape[0]
        assert len(testpixels) == testlabels_binary.shape[0]
        # trainimages_ISIC2017 = trainimages_ISIC2017.reshape(trainimages_ISIC2017.shape[0], *image_shape)

            

    @staticmethod
    def evaluate(dbpath, model_path, model_name):
        traindata, validationdata, testdata = mel.Parser.open_H5(dbpath)
        assert len(traindata['trainimages'])+len(validationdata['validationimages'])+len(testdata['testimages']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']
        assert len(traindata['trainlabels'])+len(validationdata['validationlabels'])+len(testdata['testlabels']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']
        assert len(traindata['trainids'])+len(validationdata['validationids'])+len(testdata['testids']) \
            == mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['trainimages'] + \
                mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['validationimages'] + \
                    mel.CommonData().dbNumImgs[mel.DatasetType.ISIC2017]['testimages']
        testimages_decoded = []
        for idx, img in enumerate(testdata['testimages']):
                decoded_img = img_to_array(mel.Parser.decode(img))
                decoded_img = mel.Preprocess.normalizeImg(decoded_img)
                testimages_decoded.append(decoded_img)
        testimages_decoded = np.array(testimages_decoded) # Convert list to numpy
        

        print('Testing on ISIC2017 DB')
        print(f'Evaluating {model_name} model on {mel.DatasetType.ISIC2017.name}...\n')
        model = load_model(model_path+'/'+model_name + '.hdf5')
        # model, _, _ = mel.Model.evaluate_model(
        #     model_name=model_name,
        #     model_path=model_path,
        #     target_db=mel.DatasetType.ISIC2017.name,
        #     trainimages=None,
        #     trainlabels=None,
        #     validationimages=None,
        #     validationlabels=None,
        #     testimages=testimages_decoded,
        #     testlabels=np.array(testdata['testlabels']),
        #     )
        target_network = model.layers[0].name

        test_pred, test_pred_classes = mel.Model.computing_prediction(
            model = model, model_name = model_name, target_db=mel.DatasetType.ISIC2017.name, \
            testimages = testimages_decoded)
        
        test_report = mel.Model.model_report(
            model_name = model_name, model_path=model_path, target_db=mel.DatasetType.ISIC2017.name, \
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